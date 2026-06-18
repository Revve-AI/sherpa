// sherpa/csrc/offline-zipformer2-ctc-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_CSRC_OFFLINE_ZIPFORMER2_CTC_MODEL_H_
#define SHERPA_CSRC_OFFLINE_ZIPFORMER2_CTC_MODEL_H_

#include <string>

#include "sherpa/csrc/offline-ctc-model.h"

namespace sherpa {

/** This class wraps the Zipformer2 CTC model exported from icefall.
 *
 * The torchscript module is produced by
 * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/export.py
 * (with --use-ctc 1). The exported class name is "AsrModel" and the module
 * exposes:
 *   - encoder(features, features_length) -> (encoder_out, encoder_out_lens)
 *       features:        (N, T, C) float
 *       features_length: (N,) int
 *       encoder_out:     (N, T', encoder_dim)
 *       encoder_out_lens:(N,) int  -- length after subsampling
 *   - ctc_output(encoder_out) -> log_softmax(N, T', vocab_size)
 *
 * The subsampling factor of the librispeech recipe is 4.
 */
class OfflineZipformer2CtcModel : public OfflineCtcModel {
 public:
  explicit OfflineZipformer2CtcModel(const std::string &filename,
                                     torch::Device device = torch::kCPU);

  torch::Device Device() const override { return device_; }

  int32_t SubsamplingFactor() const override { return 4; }

  /** Run encoder + ctc_output and return a tuple ivalue
   *  (log_softmax, encoder_out_lens).
   */
  torch::IValue Forward(torch::Tensor features,
                        torch::Tensor features_length) override;

  torch::Tensor GetLogSoftmaxOut(torch::IValue forward_out) const override;

  torch::Tensor GetLogSoftmaxOutLength(
      torch::IValue forward_out) const override;

 private:
  torch::Device device_;
  torch::jit::Module model_;

  // Aliases for sub-modules of model_.
  torch::jit::Module encoder_;
  torch::jit::Module ctc_output_;
};

}  // namespace sherpa

#endif  // SHERPA_CSRC_OFFLINE_ZIPFORMER2_CTC_MODEL_H_
