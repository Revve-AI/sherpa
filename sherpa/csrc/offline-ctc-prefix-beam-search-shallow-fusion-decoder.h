// sherpa/csrc/offline-ctc-prefix-beam-search-shallow-fusion-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_SHALLOW_FUSION_DECODER_H_
#define SHERPA_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_SHALLOW_FUSION_DECODER_H_

#include <vector>

#include "sherpa/csrc/context-graph.h"
#include "sherpa/csrc/offline-ctc-decoder.h"

namespace sherpa {

/** Port of icefall's `ctc_prefix_beam_search_shallow_fusion`.
 *
 * Reference:
 *   icefall/decode.py :: ctc_prefix_beam_search_shallow_fussion
 *   egs/librispeech/ASR/zipformer/ctc_decode.py (calling site)
 *
 * Compared to OfflineCtcPrefixBeamSearchDecoder this class follows the
 * icefall structure verbatim:
 *   - per-frame top-K (= beam) of ctc_output picked ONCE, reused across
 *     all live hypotheses,
 *   - each Hypothesis carries (log_prob_blank, log_prob_non_blank, lm_score),
 *     ranked by `tot_score = log_prob + lm_score` where log_prob is the
 *     logsumexp of the two streams,
 *   - hypotheses keyed by tuple(ys) and merged via logsumexp on insertion,
 *   - context bias applied via `lm_score` only when the prefix actually
 *     advances (see icefall's `update_prefix` flag), then finalized at the
 *     end via ContextGraph::Finalize.
 *
 * This implementation intentionally does NOT include NNLM / LODR shallow
 * fusion — if --lm-model is added later, plumb it through the lm_score
 * computation in the .cc.
 */
class OfflineCtcPrefixBeamSearchShallowFusionDecoder : public OfflineCtcDecoder {
 public:
  /**
   * @param beam      Number of hypotheses kept per frame AND the per-frame
   *                  top-K vocab pruning factor (matches icefall: a single
   *                  `beam` arg controls both).
   * @param blank_id  CTC blank label (almost always 0).
   */
  OfflineCtcPrefixBeamSearchShallowFusionDecoder(int32_t beam,
                                                 int32_t blank_id = 0);

  /// OfflineCtcDecoder interface (no contextual biasing).
  std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor log_prob, torch::Tensor log_prob_len,
      int32_t subsampling_factor = 1) override;

  /** Same as Decode() but with per-utterance contextual biasing.
   *
   * @param log_prob       (N, T, vocab_size) float CTC log-softmax.
   * @param log_prob_len   (N,) int valid frame counts.
   * @param context_graphs Size N. context_graphs[i] may be nullptr.
   * @param subsampling_factor Subsampling factor of the encoder.
   */
  std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor log_prob, torch::Tensor log_prob_len,
      const std::vector<ContextGraphPtr> &context_graphs,
      int32_t subsampling_factor = 1);

 private:
  int32_t beam_;
  int32_t blank_id_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_SHALLOW_FUSION_DECODER_H_
