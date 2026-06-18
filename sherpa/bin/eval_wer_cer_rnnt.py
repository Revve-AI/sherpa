#!/usr/bin/env python3
"""Evaluate WER / CER for a Zipformer + RNN-T model on a <stem>.wav + <stem>.txt
dataset directory.

Mirrors eval_wer_cer.py (CTC) but reuses the recognizer / context-encoding
helpers from offline_transducer_asr.py. Contextual biasing is fed via
--contexts-file (one phrase per line) so the same contexts.txt used for CTC
eval can be reused unchanged.

Example:
  ./sherpa/bin/eval_wer_cer_rnnt.py \
    --nn-model              ./zipformer-rnnt-exp/jit_script.pt \
    --tokens                ./zipformer-rnnt-exp/tokens.txt \
    --bpe-model             ./zipformer-rnnt-exp/bpe.model \
    --modeling-unit         bpe \
    --decoding-method       modified_beam_search \
    --num-active-paths      15 \
    --contexts-file         ./contexts.txt \
    --context-score         2.5 \
    --use-gpu               false \
    --dataset-dir           ./stt-test-bank-number \
    --batch-size            16 \
    --output-dir            ./eval_out/stt-test-bank-number-rnnt
"""
import logging
import time
from pathlib import Path
from typing import List

import torch

import sherpa  # noqa: F401  (ensures shared libs are loaded first)
from offline_transducer_asr import (
    add_decoding_args,
    add_model_args,
    add_resources_args,
    create_recognizer,
    encode_contexts,
    read_sound_files,
)
from eval_common import collect_pairs, edit_distance, normalize_text


def get_parser():
    # Build a fresh argparse so the transducer-specific decoding flags
    # (--decoding-method, --num-active-paths, --bpe-model, --modeling-unit,
    # --contexts, --context-score, --temperature, fast_beam_search opts)
    # are registered, then add the shared eval flags on top.
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_model_args(parser)
    add_decoding_args(parser)
    add_resources_args(parser)

    # Shared eval flags (kept name-compatible with eval_wer_cer.py).
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing <stem>.wav and <stem>.txt pairs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of utterances decoded together.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="If set, write hyp.txt / ref.txt / per_utt.tsv / summary.txt here.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, evaluate at most this many utterances (smoke-test).",
    )
    parser.add_argument(
        "--contexts-file",
        type=str,
        default="",
        help="Optional file with one biasing phrase per line. If set, "
        "overrides --contexts.",
    )
    return parser


def load_contexts_rnnt(args) -> List[List[int]]:
    """Build the list-of-token-IDs context list for the transducer.

    Reads phrases from --contexts-file if given, else from the
    `/`-separated --contexts string. Empty list => no biasing.
    """
    phrases: List[str] = []
    if args.contexts_file:
        cf = Path(args.contexts_file)
        if not cf.is_file():
            raise SystemExit(f"--contexts-file does not exist: {cf}")
        for line in cf.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            phrases.append(s.upper())
    elif args.contexts:
        phrases = [x.strip().upper() for x in args.contexts.split("/") if x.strip()]

    if not phrases:
        return []

    logging.info(f"Loaded {len(phrases)} contextual phrases")
    return encode_contexts(args, phrases)


def decode_batch(recognizer, wav_paths: List[Path], contexts_list,
                 sample_rate: int) -> List[str]:
    samples = read_sound_files([str(p) for p in wav_paths], sample_rate)
    streams = []
    for s in samples:
        if contexts_list:
            stream = recognizer.create_stream(contexts_list=contexts_list)
        else:
            stream = recognizer.create_stream()
        stream.accept_samples(s)
        streams.append(stream)
    recognizer.decode_streams(streams)
    return [s.result.text for s in streams]


def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"--dataset-dir does not exist: {dataset_dir}")

    pairs = collect_pairs(dataset_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No (wav, txt) pairs found under {dataset_dir}")
    logging.info(f"Found {len(pairs)} utterances under {dataset_dir}")

    # set_num_interop_threads must be called BEFORE any parallel work starts;
    # `import sherpa` already touches torch threading, so the call here may
    # raise. Treat it as a best-effort hint.
    try:
        torch.set_num_threads(args.num_threads)
    except RuntimeError as e:
        logging.warning(f"set_num_threads({args.num_threads}) ignored: {e}")
    try:
        torch.set_num_interop_threads(args.num_threads)
    except RuntimeError as e:
        logging.warning(
            f"set_num_interop_threads({args.num_threads}) ignored: {e}"
        )

    recognizer = create_recognizer(args)
    contexts_list = load_contexts_rnnt(args)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        hyp_f = open(out_dir / "hyp.txt", "w", encoding="utf-8")
        ref_f = open(out_dir / "ref.txt", "w", encoding="utf-8")
        per_utt_f = open(out_dir / "per_utt.tsv", "w", encoding="utf-8")
        per_utt_f.write("uid\twer\tcer\tref_words\thyp_words\tref\thyp\n")
    else:
        hyp_f = ref_f = per_utt_f = None

    total_word_errs = total_words = 0
    total_char_errs = total_chars = 0
    n_done = 0
    t0 = time.time()

    for batch_start in range(0, len(pairs), args.batch_size):
        batch = pairs[batch_start : batch_start + args.batch_size]
        wav_paths = [p[0] for p in batch]
        txt_paths = [p[1] for p in batch]

        try:
            hyps = decode_batch(recognizer, wav_paths, contexts_list,
                                args.sample_rate)
        except Exception as e:
            logging.warning(
                f"Batch {batch_start}-{batch_start + len(batch)} failed ({e}); "
                f"falling back to per-utt decode"
            )
            hyps = []
            for w in wav_paths:
                try:
                    hyps.append(decode_batch(recognizer, [w], contexts_list,
                                             args.sample_rate)[0])
                except Exception as ee:
                    logging.warning(f"  skip {w.name}: {ee}")
                    hyps.append("")

        for wav_path, txt_path, hyp_raw in zip(wav_paths, txt_paths, hyps):
            ref_raw = txt_path.read_text(encoding="utf-8")
            ref_n = normalize_text(ref_raw)
            hyp_n = normalize_text(hyp_raw)

            ref_words = ref_n.split()
            hyp_words = hyp_n.split()
            w_err = edit_distance(ref_words, hyp_words)
            total_word_errs += w_err
            total_words += len(ref_words)

            ref_chars = list(ref_n.replace(" ", ""))
            hyp_chars = list(hyp_n.replace(" ", ""))
            c_err = edit_distance(ref_chars, hyp_chars)
            total_char_errs += c_err
            total_chars += len(ref_chars)

            n_done += 1
            uid = wav_path.stem
            if per_utt_f:
                utt_wer = w_err / max(1, len(ref_words))
                utt_cer = c_err / max(1, len(ref_chars))
                per_utt_f.write(
                    f"{uid}\t{utt_wer:.4f}\t{utt_cer:.4f}\t"
                    f"{len(ref_words)}\t{len(hyp_words)}\t{ref_n}\t{hyp_n}\n"
                )
                hyp_f.write(f"{uid}\t{hyp_n}\n")
                ref_f.write(f"{uid}\t{ref_n}\n")

        if n_done % max(args.batch_size, 50) < args.batch_size:
            elapsed = time.time() - t0
            wer = total_word_errs / max(1, total_words)
            cer = total_char_errs / max(1, total_chars)
            logging.info(
                f"[{n_done}/{len(pairs)}] running WER={wer:.4%} "
                f"CER={cer:.4%} ({elapsed:.0f}s)"
            )

    wer = total_word_errs / max(1, total_words)
    cer = total_char_errs / max(1, total_chars)
    elapsed = time.time() - t0

    summary = (
        f"utterances : {n_done}\n"
        f"ref_words  : {total_words}\n"
        f"ref_chars  : {total_chars}\n"
        f"WER        : {wer:.4%}  ({total_word_errs} / {total_words})\n"
        f"CER        : {cer:.4%}  ({total_char_errs} / {total_chars})\n"
        f"elapsed    : {elapsed:.1f}s\n"
    )
    print(summary)
    if out_dir:
        (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
        for f in (hyp_f, ref_f, per_utt_f):
            f.close()


if __name__ == "__main__":
    torch.manual_seed(20230104)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    main()
