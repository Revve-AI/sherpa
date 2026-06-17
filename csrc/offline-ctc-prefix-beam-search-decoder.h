// sherpa/csrc/offline-ctc-prefix-beam-search-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa/cpp_api/offline-recognizer.h"
#include "sherpa/csrc/context-graph.h"
#include "sherpa/csrc/offline-ctc-decoder.h"

namespace sherpa {

/** Label-space CTC prefix beam search decoder (Hannun et al., 2014).
 *
 *   - Operates on the raw CTC log-softmax output.
 *   - Supports contextual biasing through ContextGraph: each beam state
 *     tracks an Aho-Corasick state; when a beam advances along a context
 *     phrase, ContextGraph::ForwardOneStep returns a boost added to the
 *     beam score. After decoding, ContextGraph::Finalize discounts any
 *     remaining partial-match bonus.
 *   - HLG / lexicon-based decoding is NOT applied here: use
 *     OfflineCtcOneBestDecoder if you want HLG.
 */
class OfflineCtcPrefixBeamSearchDecoder : public OfflineCtcDecoder {
 public:
  /**
   * @param num_active_paths  Beam size kept per frame (>0).
   * @param top_k             Top-K vocab pruning per frame.
   *                          0 means use the full vocabulary.
   * @param blank_id          CTC blank label (almost always 0).
   */
  OfflineCtcPrefixBeamSearchDecoder(int32_t num_active_paths, int32_t top_k,
                                    int32_t blank_id = 0);

  /// OfflineCtcDecoder interface (no contextual biasing).
  std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor log_prob, torch::Tensor log_prob_len,
      int32_t subsampling_factor = 1) override;

  /** Same as Decode() but supports per-utterance contextual biasing.
   *
   * @param log_prob          (N, T, vocab_size) float CTC log-softmax.
   * @param log_prob_len      (N,) int valid frame counts.
   * @param context_graphs    Size N. context_graphs[i] may be nullptr for
   *                          utterances without biasing.
   * @param subsampling_factor Subsampling factor of the encoder.
   */
  std::vector<OfflineCtcDecoderResult> Decode(
      torch::Tensor log_prob, torch::Tensor log_prob_len,
      const std::vector<ContextGraphPtr> &context_graphs,
      int32_t subsampling_factor = 1);

 private:
  int32_t num_active_paths_;
  int32_t top_k_;
  int32_t blank_id_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
