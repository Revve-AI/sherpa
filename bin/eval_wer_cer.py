#!/usr/bin/env python3
"""Evaluate WER / CER on a directory of <stem>.wav + <stem>.txt pairs.

Reuses the recognizer setup from offline_ctc_asr.py so flags (model, tokens,
decoding method, contexts, ...) behave identically to a single-file decode.

Example:
  ./sherpa/bin/eval_wer_cer.py \
    --nn-model              ./zipformer-ctc-exp/jit_script.pt \
    --tokens                ./zipformer-ctc-exp/tokens.txt \
    --ctc-decoding-method   prefix-beam-search \
    --ctc-num-active-paths  8 \
    --ctc-top-k             20 \
    --contexts-file         ./contexts.txt \
    --context-score         2.5 \
    --bpe-model             ./zipformer-ctc-exp/bpe.model \
    --use-gpu               false \
    --dataset-dir           ./email_evenlab_generated \
    --batch-size            16 \
    --output-dir            ./eval_out
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import torch

import sherpa  # noqa: F401  (ensures shared libs are loaded before offline_ctc_asr)
from offline_ctc_asr import (
    add_decoding_args,
    add_model_args,
    create_recognizer,
    load_contexts,
    read_sound_files,
)
from eval_common import collect_pairs, edit_distance, normalize_text
from sherpa import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_model_args(parser)
    add_decoding_args(parser)

    parser.add_argument("--use-gpu", type=str2bool, default=False)
    parser.add_argument("--normalize-samples", type=str2bool, default=True)
    parser.add_argument("--nemo-normalize", type=str, default="")

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
    return parser


def decode_batch(recognizer, wav_paths: List[Path], contexts) -> List[str]:
    samples = read_sound_files([str(p) for p in wav_paths], 16000)
    streams = []
    for s in samples:
        stream = recognizer.create_stream(contexts) if contexts else recognizer.create_stream()
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

    recognizer = create_recognizer(args)
    contexts = load_contexts(args)
    if contexts:
        logging.info(f"Loaded {len(contexts)} contextual phrases")

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
    audio_seconds_total = 0.0
    t0 = time.time()

    for batch_start in range(0, len(pairs), args.batch_size):
        batch = pairs[batch_start : batch_start + args.batch_size]
        wav_paths = [p[0] for p in batch]
        txt_paths = [p[1] for p in batch]

        try:
            hyps = decode_batch(recognizer, wav_paths, contexts)
        except Exception as e:
            logging.warning(
                f"Batch {batch_start}-{batch_start + len(batch)} failed ({e}); "
                f"falling back to per-utt decode"
            )
            hyps = []
            for w in wav_paths:
                try:
                    hyps.append(decode_batch(recognizer, [w], contexts)[0])
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
