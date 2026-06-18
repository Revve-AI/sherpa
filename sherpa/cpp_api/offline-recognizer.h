// sherpa/cpp_api/offline-recognizer.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_
#define SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa/cpp_api/fast-beam-search-config.h"
#include "sherpa/cpp_api/feature-config.h"
#include "sherpa/cpp_api/macros.h"
#include "sherpa/cpp_api/offline-stream.h"
#include "sherpa/csrc/offline-model-config.h"

namespace sherpa {

struct OfflineCtcDecoderConfig {
  // Which CTC decoder to use.
  //   - "one-best"          : k2-based HLG/CTC-topo one-best (default,
  //                           legacy behavior, supports --hlg).
  //   - "prefix-beam-search": label-space prefix beam search, supports
  //                           contextual biasing via create_stream(contexts).
  std::string method = "one-best";

  // Used only for decoding with a CTC topology
  // true to use a modified CTC topology.
  // false to use a standard CTC topology.
  bool modified = true;

  // Used only for HLG decoding
  std::string hlg;
  float lm_scale = 1.0;

  float search_beam = 20;
  float output_beam = 8;
  int32_t min_active_states = 30;
  int32_t max_active_states = 10000;

  // Used only when method == "prefix-beam-search".
  // Beam size (number of active prefixes kept per frame).
  int32_t num_active_paths = 8;

  // Used only when method == "prefix-beam-search".
  // Top-K vocab tokens considered per frame; 0 means use full vocab.
  int32_t top_k = 0;

  void Register(ParseOptions *po);
  void Validate() const;
  std::string ToString() const;
};

struct OfflineRecognizerConfig {
  /// Used only for CTC decoding.
  OfflineCtcDecoderConfig ctc_decoder_config;

  /// Config for the feature extractor
  FeatureConfig feat_config;

  FastBeamSearchConfig fast_beam_search_config;

  // TODO(fangjun): We will remmove mutable later
  mutable OfflineModelConfig model;

  /// Path to the torchscript model
  std::string nn_model;

  /// Path to tokens.txt
  std::string tokens;

  /// true to use GPU for neural network computation and decoding.
  /// false to use CPU.
  /// You can use CUDA_VISIBLE_DEVICES to control which device to use.
  /// We always use GPU 0 in the code. This also implies it supports only
  /// 1 GPU at present.
  /// Note: You have to use a CUDA version of PyTorch in order to use
  /// GPU for computation
  bool use_gpu = false;

  std::string decoding_method = "greedy_search";

  /// used only for modified_beam_search
  int32_t num_active_paths = 4;

  /// used only for modified_beam_search
  float context_score = 1.5;

  // True if the model used is trained with byte level bpe.
  bool use_bbpe = false;

  // temperature for the softmax in the joiner
  float temperature = 1.0;

  /// Constant subtracted from the blank logit before softmax in transducer
  /// greedy / modified_beam_search. Positive values discourage blank and
  /// help against deletion of repeated tokens. 0 disables.
  float blank_penalty = 0.0f;

  /// Maximum number of non-blank symbols emitted per encoder frame in
  /// transducer greedy_search. icefall default = 1. Set 2-3 to allow fast
  /// speech / repeated tokens to surface within a single frame.
  /// Ignored by modified_beam_search (which keeps the 1 sym/frame structure).
  int32_t max_sym_per_frame = 1;

  void Register(ParseOptions *po);

  void Validate() const;

  /** A string representation for debugging purpose. */
  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os,
                         const OfflineRecognizerConfig &config);

class OfflineRecognizerImpl;

class OfflineRecognizer {
 public:
  ~OfflineRecognizer();

  explicit OfflineRecognizer(const OfflineRecognizerConfig &config);

  /// Create a stream for decoding.
  std::unique_ptr<OfflineStream> CreateStream();

  /// Create a stream with contextual-biasing lists.
  std::unique_ptr<OfflineStream> CreateStream(
      const std::vector<std::vector<int32_t>> &context_list);

  /** Decode a single stream
   *
   * @param s The stream to decode.
   */
  void DecodeStream(OfflineStream *s) {
    OfflineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode a list of streams.
   *
   * @param ss Pointer to an array of streams.
   * @param n  Size of the input array.
   */
  void DecodeStreams(OfflineStream **ss, int32_t n);

 private:
  std::unique_ptr<OfflineRecognizerImpl> impl_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_OFFLINE_RECOGNIZER_H_
