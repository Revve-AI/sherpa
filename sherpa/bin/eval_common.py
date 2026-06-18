"""Shared helpers for eval_wer_cer.py and eval_wer_cer_rnnt.py.

Kept dependency-free so importing this module doesn't pull in any
torch-thread side effects from the per-model entry scripts
(offline_ctc_asr.py / offline_transducer_asr.py).
"""
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Reference data wraps English words with a language tag (e.g. "eng yahoo eng").
# Model never emits these; drop on both sides for a fair comparison.
_DROP_TOKENS = {"eng"}

# Vietnamese tone-mark placement: modern style puts the tone on the first vowel
# of open -oa / -oe / -uy syllables ("thùy"); older style puts it on the second
# ("thuỳ"). Both appear in ref vs hyp. Normalize old -> modern at end-of-word.
_VN_TONE_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
_VN_TONE_RE = re.compile("(" + "|".join(map(re.escape, _VN_TONE_MAP)) + r")(?=\s|$)")


def _merge_spelled_letters(words: List[str]) -> List[str]:
    # ["a", "b", "c"] -> ["abc"]; isolated single letters are left alone.
    out: List[str] = []
    buf: List[str] = []
    for w in words:
        if len(w) == 1 and w.isalpha():
            buf.append(w)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(w)
    if buf:
        out.append("".join(buf))
    return out


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _VN_TONE_RE.sub(lambda m: _VN_TONE_MAP[m.group(1)], s)
    words = [w for w in s.split() if w not in _DROP_TOKENS]
    return " ".join(_merge_spelled_letters(words))


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def collect_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for wav in sorted(dataset_dir.glob("*.wav")):
        if wav.stat().st_size == 0:
            continue
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        pairs.append((wav, txt))
    return pairs
