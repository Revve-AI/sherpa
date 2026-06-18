// sherpa/csrc/offline-zipformer2-ctc-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa/csrc/offline-zipformer2-ctc-model.h"

#include <vector>

#include "sherpa/cpp_api/macros.h"

namespace sherpa {

OfflineZipformer2CtcModel::OfflineZipformer2CtcModel(
    const std::string &filename, torch::Device device /*= torch::kCPU*/)
    : device_(device) {
  model_ = torch::jit::load(filename, device);
  model_.eval();

  encoder_ = model_.attr("encoder").toModule();
  ctc_output_ = model_.attr("ctc_output").toModule();
}

torch::IValue OfflineZipformer2CtcModel::Forward(
    torch::Tensor features, torch::Tensor features_length) {
  InferenceMode no_grad;

  // encoder.forward returns (encoder_out, encoder_out_lens).
  auto enc_out = encoder_.run_method("forward", features.to(device_),
                                     features_length.to(device_));

  auto enc_tuple = enc_out.toTuple();
  torch::Tensor encoder_out = enc_tuple->elements()[0].toTensor();
  torch::Tensor encoder_out_lens = enc_tuple->elements()[1].toTensor();

  // ctc_output is a Sequential(Dropout, Linear, LogSoftmax) module, so calling
  // its forward already yields log-probabilities of shape (N, T', vocab_size).
  torch::Tensor log_prob =
      ctc_output_.run_method("forward", encoder_out).toTensor();

  std::vector<torch::IValue> ans = {log_prob, encoder_out_lens};
  return torch::ivalue::Tuple::create(ans);
}

torch::Tensor OfflineZipformer2CtcModel::GetLogSoftmaxOut(
    torch::IValue forward_out) const {
  return forward_out.toTuple()->elements()[0].toTensor();
}

torch::Tensor OfflineZipformer2CtcModel::GetLogSoftmaxOutLength(
    torch::IValue forward_out) const {
  return forward_out.toTuple()->elements()[1].toTensor();
}

}  // namespace sherpa
