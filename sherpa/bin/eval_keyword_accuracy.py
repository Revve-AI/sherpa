#!/usr/bin/env python3
"""Keyword accuracy of an ASR run, judged against GPT-extracted keywords.

Pipeline:
  1. Read ref.txt / hyp.txt (each line: "<uid>\t<text>") from an eval_out dir.
  2. Ask OpenAI gpt-5.2 to pull the salient keywords out of every *ref* line
     (names, numbers, domain terms — the tokens that actually carry meaning).
  3. A keyword counts as "hit" when its normalized form appears as a contiguous
     token run in the normalized *hyp* line.
  4. keyword_accuracy = total hits / total keywords (micro-averaged).

Extraction is cached to <output>/keywords_cache.json keyed by uid, so re-runs
are free and only newly-seen utterances hit the API.

Example:
  export OPENAI_API_KEY=sk-...
  ./sherpa/bin/eval_keyword_accuracy.py \
    --eval-dir ./eval_out/stt-test-callcenter-noised-denoised-rnnt-old
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Reuse the exact normalization the WER/CER eval uses, so a keyword "hit" here
# means the same thing as a correct token there.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import normalize_text  # noqa: E402

MODEL = "gpt-5.2"
BATCH = 20  # ref lines per API request
WORKERS = 8  # concurrent requests

def load_dotenv(start: Path) -> None:
    """Populate os.environ from the nearest .env up the tree (no overwrite)."""
    for d in [start, *start.parents]:
        f = d / ".env"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        return


SYSTEM_PROMPT = (
    "Bạn trích xuất từ khóa quan trọng từ câu thoại tổng đài tiếng Việt. "
    "Từ khóa là những token mang thông tin: tên riêng, số (điện thoại, tiền, "
    "mã), thuật ngữ nghiệp vụ, sản phẩm, dịch vụ, hành động chính. Bỏ qua từ "
    "đệm, hư từ, từ chào hỏi. "
    "QUAN TRỌNG: từ khóa phải GIỮ NGUYÊN VĂN dạng nói (spoken form) đúng y như "
    "trong câu — copy chính xác các chữ liên tiếp từ câu gốc. KHÔNG chuyển số "
    "đọc thành chữ số (giữ 'mười lăm', KHÔNG đổi thành '15'), KHÔNG viết tắt, "
    "KHÔNG sửa chính tả, KHÔNG chuẩn hóa sang dạng viết. "
    "Mỗi từ khóa có thể là cụm nhiều từ liên tiếp. "
    "Nếu câu không có từ khóa nào, trả về danh sách rỗng."
)


def contains_run(needle: str, haystack_tokens: List[str]) -> bool:
    """True if `needle`'s normalized tokens appear as a contiguous run."""
    kw = normalize_text(needle).split()
    if not kw:
        return False
    n = len(kw)
    return any(haystack_tokens[i : i + n] == kw for i in range(len(haystack_tokens) - n + 1))


def read_tsv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uid, _, text = line.partition("\t")
        out[uid] = text
    return out


def extract_batch(client, texts: List[str]) -> Dict[str, List[str]]:
    """texts: list of distinct ref strings -> {text: [keyword, ...]}.

    Tagged by index in the request so the model can't scramble the mapping.
    """
    lines = [{"i": i, "text": t} for i, t in enumerate(texts)]
    user = (
        "Trích xuất từ khóa cho từng câu dưới đây. "
        "Trả về JSON: {\"results\": [{\"i\": ..., \"keywords\": [...]}, ...]}.\n\n"
        + json.dumps(lines, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    out: Dict[str, List[str]] = {}
    for r in data.get("results", []):
        src = texts[int(r["i"])]
        src_tokens = normalize_text(src).split()
        # Keep only keywords that are verbatim contiguous runs of the ref
        # (spoken form) — drops any the model rewrote into written form.
        out[src] = [str(k) for k in r.get("keywords", []) if contains_run(str(k), src_tokens)]
    return out


def get_keywords(client, ref: Dict[str, str], cache_path: Path) -> Dict[str, List[str]]:
    # Cache is keyed by the ref *text*, not uid: keywords for an identical
    # sentence are reused across datasets/runs, and a changed ref re-extracts.
    cache: Dict[str, List[str]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    # Distinct uncached texts only -> never pay twice for the same sentence.
    todo = sorted({t for t in ref.values() if t not in cache})
    if todo:
        print(f"extracting keywords for {len(todo)} new sentences via {MODEL} "
              f"({len(ref) - len(todo)} already cached) ...")
        batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(extract_batch, client, b): b for b in batches}
            for fut in as_completed(futs):
                try:
                    cache.update(fut.result())
                except Exception as e:  # keep partial progress, retry failed next run
                    print(f"  batch failed: {e}", file=sys.stderr)
                done += 1
                if done % 10 == 0 or done == len(batches):
                    cache_path.write_text(
                        json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8"
                    )
                    print(f"  {done}/{len(batches)} batches, {len(cache)} sentences cached")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")

    # Project text-keyed cache back onto uids for the caller.
    return {uid: cache[t] for uid, t in ref.items() if t in cache}


def keyword_hit(keyword: str, hyp_tokens: List[str]) -> bool:
    return contains_run(keyword, hyp_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, type=Path,
                    help="dir containing ref.txt and hyp.txt")
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--hyp", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="per-utterance TSV output (default: <eval-dir>/keyword_acc.tsv)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="shared keyword cache file. Point several eval dirs with the "
                         "same ref.txt at one path to extract keywords only once "
                         "(default: <eval-dir>/keywords_cache.json)")
    args = ap.parse_args()

    ref_path = args.ref or args.eval_dir / "ref.txt"
    hyp_path = args.hyp or args.eval_dir / "hyp.txt"
    out_path = args.out or args.eval_dir / "keyword_acc.tsv"
    cache_path = args.cache or args.eval_dir / "keywords_cache.json"

    load_dotenv(Path(__file__).resolve().parent)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (checked env and .env)")

    from openai import OpenAI

    client = OpenAI()
    ref = read_tsv(ref_path)
    hyp = read_tsv(hyp_path)

    t0 = time.time()
    keywords = get_keywords(client, ref, cache_path)

    total_kw = hit_kw = 0
    rows: List[str] = ["uid\tn_kw\tn_hit\tacc\tmissed"]
    for uid, kws in keywords.items():
        if uid not in hyp:
            continue
        hyp_tokens = normalize_text(hyp[uid]).split()
        hits = [k for k in kws if keyword_hit(k, hyp_tokens)]
        missed = [k for k in kws if k not in hits]
        total_kw += len(kws)
        hit_kw += len(hits)
        acc = len(hits) / len(kws) if kws else 1.0
        rows.append(f"{uid}\t{len(kws)}\t{len(hits)}\t{acc:.4f}\t{' | '.join(missed)}")

    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    overall = hit_kw / total_kw if total_kw else 0.0
    summary = (
        f"model            : {MODEL}\n"
        f"utterances       : {sum(1 for u in keywords if u in hyp)}\n"
        f"keywords         : {total_kw}\n"
        f"keyword hits     : {hit_kw}\n"
        f"keyword accuracy : {overall * 100:.4f}%  ({hit_kw} / {total_kw})\n"
        f"elapsed          : {time.time() - t0:.1f}s\n"
    )
    (args.eval_dir / "keyword_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"per-utt -> {out_path}")


if __name__ == "__main__":
    main()
