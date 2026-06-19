#!/usr/bin/env python3
# Copyright (c)  2023  Xiaomi Corporation

"""
A standalone script for offline (i.e., non-streaming) speech recognition.

This file decodes files without the need to start a server and a client.

Please refer to
https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc.html
for pre-trained models to download.

Usage:
(1) Use icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09
cd icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09

git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "data/lang_bpe_500/HLG.pt"

cd /path/to/sherpa

(a) Decoding with H

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
  --tokens ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/tokens.txt \
  --use-gpu false \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

(b) Decoding with HLG

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/exp/cpu_jit.pt \
  --tokens ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/tokens.txt \
  --HLG ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/data/lang_bpe_500/HLG.pt \
  --lm-scale 0.9 \
  --use-gpu false \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1089-134686-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0001.wav \
  ./icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09/test_wavs/1221-135766-0002.wav

(2) Use wenet-english-model

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wenet-english-model
cd wenet-english-model
git lfs pull --include "final.zip"

cd /path/to/sherpa

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./wenet-english-model/final.zip \
  --tokens ./wenet-english-model/units.txt \
  --use-gpu false \
  --normalize-samples false \
  ./wenet-english-model/test_wavs/1089-134686-0001.wav \
  ./wenet-english-model/test_wavs/1221-135766-0001.wav \
  ./wenet-english-model/test_wavs/1221-135766-0002.wav

(3) Use wav2vec2.0-torchaudio

cd /path/to/sherpa

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/wav2vec2.0-torchaudio
cd wav2vec2.0-torchaudio
git lfs pull --include "wav2vec2_asr_base_10m.pt"

cd /path/to/sherpa

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./wav2vec2.0-torchaudio/wav2vec2_asr_base_10m.pt \
  --tokens ./wav2vec2.0-torchaudio/tokens.txt \
  --use-gpu false \
  ./wav2vec2.0-torchaudio/test_wavs/1089-134686-0001.wav \
  ./wav2vec2.0-torchaudio/test_wavs/1221-135766-0001.wav \
  ./wav2vec2.0-torchaudio/test_wavs/1221-135766-0002.wav

(4) Use icefall Zipformer2 + CTC (HLG, 1best)

cd /path/to/sherpa

# Export `cpu_jit.pt` with icefall's zipformer/export.py using --use-ctc 1
# (the resulting torchscript class name is "AsrModel"). Then build/compile
# an HLG.pt for your lang dir (icefall: local/compile_hlg.py).

# (a) Decoding with H (CTC topo only — no HLG)

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./zipformer-ctc-exp/cpu_jit.pt \
  --tokens   ./zipformer-ctc-exp/data/lang_bpe_500/tokens.txt \
  --use-gpu  false \
  ./test_wavs/1089-134686-0001.wav \
  ./test_wavs/1221-135766-0001.wav

# (b) Decoding with HLG (1-best)

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./zipformer-ctc-exp/cpu_jit.pt \
  --tokens   ./zipformer-ctc-exp/data/lang_bpe_500/tokens.txt \
  --HLG      ./zipformer-ctc-exp/data/lang_bpe_500/HLG.pt \
  --lm-scale 0.9 \
  --use-gpu  false \
  ./test_wavs/1089-134686-0001.wav \
  ./test_wavs/1221-135766-0001.wav

(4c) Contextual biasing with prefix-beam-search (no HLG)

# Bias the decoder toward a custom list of (possibly OOV) phrases. The
# search runs in BPE label space — no HLG/lexicon is consulted, so genuine
# OOV words can be added at runtime.

# Step 1: prepare contexts.txt — one phrase per line. Two accepted formats
# per line:
#   - plain text  : tokenized at runtime via --bpe-model (sentencepiece)
#   - "INT INT .."  : pre-tokenized token IDs (no --bpe-model needed)
#
# Example contexts.txt:
#   NGUYỄN TRUNG TRỰC
#   SHERPA
#   ZIPFORMER

./sherpa/bin/offline_ctc_asr.py \
  --nn-model           ./zipformer-ctc-exp/cpu_jit.pt \
  --tokens             ./zipformer-ctc-exp/data/lang_bpe_500/tokens.txt \
  --ctc-decoding-method prefix-beam-search \
  --ctc-num-active-paths 8 \
  --contexts-file      ./contexts.txt \
  --bpe-model          ./zipformer-ctc-exp/data/lang_bpe_500/bpe.model \
  --context-score      2.0 \
  --use-gpu false \
  test.wav

# Option A (alternative, offline): rebuild HLG with extra lexicon entries.
# This keeps you on the --HLG path but requires:
#   1) Add new words + pronunciation (BPE pieces) to lang_bpe_500/lexicon.txt
#      (use G2P or eSpeak for unknown words).
#   2) Re-run icefall: `python local/compile_hlg.py --lang-dir data/lang_bpe_500`
#   3) Re-point --HLG at the new HLG.pt. No code change needed in sherpa.
# Recommended when the context list is static; prefix-beam-search above is
# better for runtime-mutable lists or true OOV.

(5) Use NeMo CTC models

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-nemo-ctc-en-citrinet-512
cd sherpa-nemo-ctc-en-citrinet-512
git lfs pull --include "model.pt"

cd /path/to/sherpa

./sherpa/bin/offline_ctc_asr.py \
  --nn-model ./sherpa-nemo-ctc-en-citrinet-512/model.pt
  --tokens ./sherpa-nemo-ctc-en-citrinet-512/tokens.txt \
  --use-gpu false \
  --nemo-normalize per_feature \
  ./sherpa-nemo-ctc-en-citrinet-512/test_wavs/0.wav \
  ./sherpa-nemo-ctc-en-citrinet-512/test_wavs/1.wav \
  ./sherpa-nemo-ctc-en-citrinet-512/test_wavs/2.wav
"""  # noqa
import argparse
import logging
from pathlib import Path
from typing import List

import torch
import torchaudio

import sherpa
from sherpa import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_model_args(parser)
    add_decoding_args(parser)

    parser.add_argument(
        "--use-gpu",
        type=str2bool,
        default=False,
        help="""True to use GPU. It always selects GPU 0. You can use the
        environement variable CUDA_VISIBLE_DEVICES to control which GPU
        is mapped to GPU 0.
        """,
    )

    parser.add_argument(
        "--normalize-samples",
        type=str2bool,
        default=True,
        help="""If your model was trained using features computed
        from samples in the range `[-32768, 32767]`, then please set
        this flag to False. For instance, if you use models from WeNet,
        please set it to False.
        """,
    )

    parser.add_argument(
        "--nemo-normalize",
        type=str,
        default="",
        help="""Used only for models from NeMo.
        Leave it to empty if the preprocessor of the model does not use
        normalization. Current supported value is "per_feature".
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. ",
    )

    return parser


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--nn-model",
        type=str,
        help="""The torchscript model. Please refer to
        https://k2-fsa.github.io/sherpa/cpp/pretrained_models/offline_ctc/index.html
        for a list of pre-trained models to download.
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )


def add_decoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--HLG",
        type=str,
        help="""Optional. If empty, we use an H graph for decoding.
        If not empty, it is the filename of HLG.pt and we will
        use it for decoding""",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=1.0,
        help="""
        Used only when --HLG is not empty. It specifies the scale
        for HLG.scores
        """,
    )

    parser.add_argument(
        "--modified",
        type=bool,
        default=True,
        help="""Used only when --HLG is empty. True to use a modified
        CTC topology. False to use a standard CTC topology.
        Please refer to https://k2-fsa.github.io/k2/python_api/api.html#ctc-topo
        for the differences between standard and modified CTC topology.
        """,
    )

    parser.add_argument(
        "--search-beam",
        type=float,
        default=20.0,
        help="""Decoding beam, e.g. 20.  Smaller is faster, larger is
        more exact (less pruning). This is the default value;
        it may be modified by `min_active_states` and
        `max_active_states`.
        """,
    )

    parser.add_argument(
        "--output-beam",
        type=float,
        default=8.0,
        help="""Beam to prune output, similar to lattice-beam in Kaldi.
        Relative to the best path of output.
        """,
    )

    parser.add_argument(
        "--min-active-states",
        type=int,
        default=30,
        help="""Minimum number of FSA states that are allowed to
         be active on any given frame for any given
        intersection/composition task. This is advisory,
        in that it will try not to have fewer than this
        number active. Set it to zero if there is no
        constraint.""",
    )

    parser.add_argument(
        "--max-active-states",
        type=int,
        default=10000,
        help="""Maximum number of FSA states that are allowed to
        be active on any given frame for any given
        intersection/composition task. This is advisory,
        in that it will try not to exceed that but may
        not always succeed. You can use a very large
        number if no constraint is needed.""",
    )

    parser.add_argument(
        "--ctc-decoding-method",
        type=str,
        default="one-best",
        choices=[
            "one-best",
            "prefix-beam-search",
            "prefix-beam-search-shallow-fusion",
        ],
        help="""CTC decoding method.
        - one-best: k2-based HLG/CTC-topo one-best (default, supports --HLG).
        - prefix-beam-search: label-space prefix beam search with optional
          contextual biasing via --contexts-file (does NOT use HLG).
        - prefix-beam-search-shallow-fusion: port of icefall's
          `ctc_prefix_beam_search_shallow_fussion` (decode.py). Same
          contextual biasing path as `prefix-beam-search`; NNLM/LODR LM
          fusion is currently a no-op (placeholder for future --lm-model
          support). `--ctc-num-active-paths` controls both beam size AND
          per-frame top-K vocab (icefall uses a single `beam` arg).
        """,
    )

    parser.add_argument(
        "--ctc-num-active-paths",
        type=int,
        default=8,
        help="Beam size for --ctc-decoding-method=prefix-beam-search.",
    )

    parser.add_argument(
        "--ctc-top-k",
        type=int,
        default=0,
        help="""Top-K vocab pruning per frame for prefix-beam-search.
        0 means use full vocab.""",
    )

    parser.add_argument(
        "--contexts-file",
        type=str,
        default="",
        help="""Path to a file of contextual-biasing phrases (one per line).
        Used only when --ctc-decoding-method=prefix-beam-search. Each line
        is either:
          (a) a space-separated list of token IDs (pre-tokenized), or
          (b) plain text — in which case --bpe-model must be given so we can
              tokenize via sentencepiece.
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="",
        help="""Path to the sentencepiece BPE model used during training.
        Required only when --contexts-file contains plain text. Install via
        `pip install sentencepiece`.""",
    )

    parser.add_argument(
        "--context-score",
        type=float,
        default=1.5,
        help="""Per-token bonus added when a beam advances along a context
        phrase. Used only when --contexts-file is non-empty.""",
    )


def check_args(args):
    if not Path(args.nn_model).is_file():
        raise ValueError(f"{args.nn_model} does not exist")

    if not Path(args.tokens).is_file():
        raise ValueError(f"{args.tokens} does not exist")

    if args.HLG:
        assert Path(args.HLG).is_file(), f"{args.HLG} does not exist"

    if args.contexts_file:
        if args.ctc_decoding_method not in (
            "prefix-beam-search",
            "prefix-beam-search-shallow-fusion",
        ):
            raise ValueError(
                "--contexts-file requires --ctc-decoding-method in "
                "{prefix-beam-search, prefix-beam-search-shallow-fusion}"
            )
        if not Path(args.contexts_file).is_file():
            raise ValueError(f"{args.contexts_file} does not exist")
        if args.bpe_model and not Path(args.bpe_model).is_file():
            raise ValueError(f"{args.bpe_model} does not exist")

    assert len(args.sound_files) > 0, args.sound_files
    for f in args.sound_files:
        if not Path(f).is_file():
            raise ValueError(f"{f} does not exist")


def load_contexts(args) -> List[List[int]]:
    """Read --contexts-file into a list of token-ID lists.

    Each non-empty / non-comment line is one biasing phrase. Accepted formats:
      (1) Pre-tokenized: space-separated integer token IDs.
      (2) Plain text: tokenized via tokens.txt lookup. Each whitespace-split
          word is looked up as "▁<word>" (icefall BPE word-initial form),
          then as "<word>". If every word resolves, no extra dependency
          is needed — handy when context phrases are common in-vocab words.
      (3) Plain text + --bpe-model: tokenize via sentencepiece. Required
          when any word is OOV w.r.t. tokens.txt.
    Detection is per-file: (1) wins if every line parses as ints; otherwise
    we try (2), and fall back to (3) only on unresolved words.
    """
    if not args.contexts_file:
        return []

    raw_lines: List[str] = []
    with open(args.contexts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_lines.append(line)

    if not raw_lines:
        return []

    def parse_int_line(s: str):
        try:
            return [int(x) for x in s.split()]
        except ValueError:
            return None

    parsed = [parse_int_line(l) for l in raw_lines]
    if all(p is not None for p in parsed):
        return parsed  # format (1)

    # Build symbol -> id map from tokens.txt (format 2 / fallback for 3).
    sym2id = {}
    with open(args.tokens, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) < 2:
                continue
            sym2id[parts[0]] = int(parts[1])

    def tokenize_with_table(phrase: str):
        ids = []
        unknown = []
        for w in phrase.split():
            key = "▁" + w  # ▁ word-initial marker used by icefall BPE
            if key in sym2id:
                ids.append(sym2id[key])
            elif w in sym2id:
                ids.append(sym2id[w])
            else:
                unknown.append(w)
        return ids, unknown

    results: List[List[int]] = []
    unresolved_lines: List[int] = []
    for i, line in enumerate(raw_lines):
        ids, unknown = tokenize_with_table(line)
        if unknown:
            unresolved_lines.append(i)
            results.append(None)
        else:
            results.append(ids)

    if not unresolved_lines:
        return results  # format (2)

    # Format (3): sentencepiece tokenization — closest match to training.
    if args.bpe_model:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)
        for i in unresolved_lines:
            results[i] = sp.encode(raw_lines[i], out_type=int)
        return results

    # Format (3b): greedy longest-prefix-match fallback using tokens.txt.
    # No merge rules → may differ from training-time BPE segmentation; usually
    # close enough for biasing purposes. Pass --bpe-model for exact match.
    def greedy_tokenize(phrase: str):
        ids = []
        for word in phrase.split():
            rem = "▁" + word  # icefall word-initial marker
            while rem:
                match_id = None
                match_len = 0
                for L in range(len(rem), 0, -1):
                    cand = rem[:L]
                    if cand in sym2id:
                        match_id = sym2id[cand]
                        match_len = L
                        break
                if match_id is None:
                    return None  # even single char not in vocab
                ids.append(match_id)
                rem = rem[match_len:]
        return ids

    failures = []
    for i in unresolved_lines:
        ids = greedy_tokenize(raw_lines[i])
        if ids is None:
            failures.append(raw_lines[i])
        else:
            results[i] = ids

    if failures:
        raise ValueError(
            f"Cannot tokenize {len(failures)} context line(s) even with "
            f"greedy fallback. Pass --bpe-model <bpe.model> for exact "
            f"BPE encoding. Examples: {failures[:3]}"
        )

    logging.warning(
        f"{len(unresolved_lines)} context phrase(s) tokenized via greedy "
        f"longest-prefix fallback (no --bpe-model). Segmentation may differ "
        f"from training BPE; pass --bpe-model for exact match."
    )
    return results


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        if sample_rate != expected_sample_rate:
            wave = torchaudio.functional.resample(
                wave,
                orig_freq=sample_rate,
                new_freq=expected_sample_rate,
            )

        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


def create_recognizer(args):
    feat_config = sherpa.FeatureConfig()

    # feat_config.fbank_opts.frame_opts.samp_freq = 16000
    # feat_config.fbank_opts.mel_opts.num_bins = 80
    # feat_config.fbank_opts.mel_opts.high_freq = -400
    # feat_config.fbank_opts.frame_opts.dither = 0

    feat_config.normalize_samples = args.normalize_samples
    feat_config.nemo_normalize = args.nemo_normalize

    ctc_decoder_config = sherpa.OfflineCtcDecoderConfig(
        method=args.ctc_decoding_method,
        hlg=args.HLG if args.HLG else "",
        lm_scale=args.lm_scale,
        modified=args.modified,
        search_beam=args.search_beam,
        output_beam=args.output_beam,
        min_active_states=args.min_active_states,
        max_active_states=args.max_active_states,
        num_active_paths=args.ctc_num_active_paths,
        top_k=args.ctc_top_k,
    )

    config = sherpa.OfflineRecognizerConfig(
        nn_model=args.nn_model,
        tokens=args.tokens,
        use_gpu=args.use_gpu,
        feat_config=feat_config,
        ctc_decoder_config=ctc_decoder_config,
        context_score=args.context_score,
    )

    recognizer = sherpa.OfflineRecognizer(config)

    return recognizer


def main():
    args = get_parser().parse_args()
    logging.info(vars(args))
    check_args(args)

    recognizer = create_recognizer(args)
    sample_rate = 16000

    samples: List[torch.Tensor] = read_sound_files(
        args.sound_files,
        sample_rate,
    )

    contexts = load_contexts(args)
    if contexts:
        logging.info(
            f"Loaded {len(contexts)} contextual phrases from {args.contexts_file}"
        )

    streams: List[sherpa.OfflineStream] = []
    for s in samples:
        if contexts:
            stream = recognizer.create_stream(contexts)
        else:
            stream = recognizer.create_stream()
        stream.accept_samples(s)
        streams.append(stream)

    recognizer.decode_streams(streams)
    for filename, stream in zip(args.sound_files, streams):
        print(f"{filename}\n{stream.result}")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
"""
// Use the following in C++
torch::jit::getExecutorMode() = false;
torch::jit::getProfilingMode() = false;
torch::jit::setGraphExecutorOptimize(false);
"""

if __name__ == "__main__":
    torch.manual_seed(20230104)
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
