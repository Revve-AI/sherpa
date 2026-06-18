// sherpa/csrc/offline-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa/csrc/offline-transducer-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/macros.h"
#include "torch/all.h"

namespace sherpa {

/**
 * Construct the decoder input from the current hypothesis.
 *
 * @param hyps  A list-of-list of token IDs containing the current decoding
 *              results. Its length is `batch_size`
 * @param decoder_input A 2-D tensor of shape (batch_size, context_size).
 */
static void BuildDecoderInput(
    const std::vector<OfflineTransducerDecoderResult> &r,
    torch::Tensor *decoder_input) {
  int32_t batch_size = decoder_input->size(0);
  int32_t context_size = decoder_input->size(1);
  int64_t *p = decoder_input->data_ptr<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    auto start = r[i].tokens.end() - context_size;
    auto end = r[i].tokens.end();
    std::copy(start, end, p);
    p += context_size;
  }
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerGreedySearchDecoder::Decode(torch::Tensor encoder_out,
                                             torch::Tensor encoder_out_length,
                                             OfflineStream **ss /*= nullptr*/,
                                             int32_t n /*= 0*/) {
  InferenceMode no_grad;

  TORCH_CHECK(encoder_out.dim() == 3, "encoder_out.dim() is ",
              encoder_out.dim(), "Expected value is 3");
  TORCH_CHECK(encoder_out.scalar_type() == torch::kFloat,
              "encoder_out.scalar_type() is ", encoder_out.scalar_type());

  TORCH_CHECK(encoder_out_length.dim() == 1, "encoder_out_length.dim() is",
              encoder_out_length.dim());
  TORCH_CHECK(encoder_out_length.scalar_type() == torch::kLong,
              "encoder_out_length.scalar_type() is ",
              encoder_out_length.scalar_type());

  TORCH_CHECK(encoder_out_length.device().is_cpu());

  torch::Device device = model_->Device();

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out,
                                                  encoder_out_length,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);

  int32_t blank_id = 0;  // hard-code
  int32_t context_size = model_->ContextSize();

  int32_t N = encoder_out_length.size(0);

  std::vector<OfflineTransducerDecoderResult> results(N);

  std::vector<int32_t> padding(context_size, -1);
  padding.back() = blank_id;

  for (auto &r : results) {
    // We will remove the padding at the end
    r.tokens = padding;
  }

  auto decoder_input =
      torch::full({N, context_size}, -1,
                  torch::dtype(torch::kLong)
                      .memory_format(torch::MemoryFormat::Contiguous));

  // set the last column to blank_id, i.e., decoder_input[:, -1] = blank_id
  decoder_input.index({torch::indexing::Slice(), -1}) = blank_id;

  // its shape is (N, 1, joiner_dim)
  auto decoder_out = model_->RunDecoder(decoder_input.to(device));

  using torch::indexing::Slice;
  auto batch_sizes_accessor = packed_seq.batch_sizes().accessor<int64_t, 1>();

  int32_t max_T = packed_seq.batch_sizes().numel();

  int32_t offset = 0;
  for (int32_t t = 0; t != max_T; ++t) {
    int32_t cur_batch_size = batch_sizes_accessor[t];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = packed_seq.data().index({Slice(start, end)});
    offset = end;

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out is of shape (cur_batch_size, 1, 1, joiner_dim)
    if (cur_batch_size < decoder_out.size(0)) {
      decoder_out = decoder_out.index({Slice(0, cur_batch_size)});
    }

    // Inner loop: emit up to max_sym_per_frame_ non-blank symbols at this
    // encoder frame. Per-utterance counter `sym_in_frame[k]` enforces the
    // cap; when no hyp emits non-blank in a step, we break to t+1.
    int32_t max_steps = std::max(1, max_sym_per_frame_);
    std::vector<int32_t> sym_in_frame(cur_batch_size, 0);

    for (int32_t step = 0; step < max_steps; ++step) {
      auto logits =
          model_->RunJoiner(cur_encoder_out, decoder_out.unsqueeze(1));
      // logits shape: (cur_batch_size, 1, 1, vocab_size)
      logits = logits.squeeze(1).squeeze(1);

      // Discourage blank emission if requested. greedy uses raw logits;
      // subtracting a constant from blank's logit is the standard knob.
      if (blank_penalty_ > 0.0f) {
        logits.index({torch::indexing::Slice(), blank_id}) -= blank_penalty_;
      }

      auto max_indices = logits.argmax(/*dim*/ -1).cpu();
      auto max_indices_accessor = max_indices.accessor<int64_t, 1>();

      bool emitted_this_step = false;
      for (int32_t k = 0; k != cur_batch_size; ++k) {
        if (sym_in_frame[k] >= max_steps) continue;
        auto index = max_indices_accessor[k];
        if (index != blank_id) {
          results[k].tokens.push_back(index);
          results[k].timestamps.push_back(t);
          sym_in_frame[k] += 1;
          emitted_this_step = true;
        }
      }

      if (!emitted_this_step) break;  // everyone blanked → next encoder frame

      // Re-evaluate joiner next step only if we might still emit more.
      if (step + 1 < max_steps) {
        BuildDecoderInput(results, &decoder_input);
        decoder_out = model_->RunDecoder(decoder_input.to(device));
      } else {
        // Final emission of this frame: update decoder_out for next t.
        BuildDecoderInput(results, &decoder_input);
        decoder_out = model_->RunDecoder(decoder_input.to(device));
      }
    }
  }  // for (int32_t t = 0; t != max_T; ++t) {

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  std::vector<OfflineTransducerDecoderResult> ans(N);

  for (int32_t i = 0; i != N; ++i) {
    int32_t k = unsorted_indices_accessor[i];
    torch::ArrayRef<int32_t> arr(results[k].tokens);
    ans[i].tokens = arr.slice(context_size).vec();
    ans[i].timestamps = std::move(results[k].timestamps);
  }

  return ans;
}

}  // namespace sherpa
