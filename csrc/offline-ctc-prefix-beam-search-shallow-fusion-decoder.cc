// sherpa/csrc/offline-ctc-prefix-beam-search-shallow-fusion-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation
//
// Port of icefall's `ctc_prefix_beam_search_shallow_fussion`:
//   icefall/decode.py
//   egs/librispeech/ASR/zipformer/ctc_decode.py (caller)
//
// Algorithm mapping (icefall name -> here):
//   HypothesisList               -> HypothesisList (anon namespace below)
//   Hypothesis.ys                -> Hypothesis::ys
//   Hypothesis.log_prob_blank    -> Hypothesis::log_prob_blank
//   Hypothesis.log_prob_non_blank-> Hypothesis::log_prob_non_blank
//   Hypothesis.lm_score          -> Hypothesis::lm_score (context only here)
//   tot_score = log_prob+lm_score-> Hypothesis::TotScore()
//   _step_worker                 -> StepWorker()
//
// NOTE: NNLM / LODR shallow fusion is intentionally omitted (the user
// asked for the icefall structure without LM). To add it later, accumulate
// the LM contribution into `lm_score` inside the `update_prefix` branch
// of StepWorker(), mirroring icefall's nnlm_scale / LODR_lm_scale blocks.

#include "sherpa/csrc/offline-ctc-prefix-beam-search-shallow-fusion-decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "sherpa/cpp_api/macros.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float LogAdd(float a, float b) {
  if (a == kNegInf) return b;
  if (b == kNegInf) return a;
  float m = std::max(a, b);
  return m + std::log1pf(std::expf(-std::fabs(a - b)));
}

struct Hypothesis {
  std::vector<int32_t> ys;
  float log_prob_blank = kNegInf;
  float log_prob_non_blank = kNegInf;
  float lm_score = 0.0f;  // context-bias accumulator (no LM here).
  const ContextState *context_state = nullptr;
  std::vector<int32_t> timestamps;

  float LogProb() const { return LogAdd(log_prob_blank, log_prob_non_blank); }
  float TotScore() const { return LogProb() + lm_score; }
};

// Merge-on-insert hypothesis collection keyed by tuple(ys).
// Mirrors icefall.utils.HypothesisList.add() which logsumexps both streams
// on key collision.
class HypothesisList {
 public:
  void Add(Hypothesis &&h) {
    auto it = map_.find(h.ys);
    if (it == map_.end()) {
      map_.emplace(h.ys, std::move(h));
      return;
    }
    Hypothesis &old = it->second;
    old.log_prob_blank = LogAdd(old.log_prob_blank, h.log_prob_blank);
    old.log_prob_non_blank =
        LogAdd(old.log_prob_non_blank, h.log_prob_non_blank);
    // lm_score / context_state / timestamps: keep the *higher-scoring*
    // contributor's data. Since both share the same ys, picking either
    // is correct for context_state; we prefer the new one if it improves
    // tot_score (mirrors icefall add() comment).
    if (h.lm_score > old.lm_score) {
      old.lm_score = h.lm_score;
      old.context_state = h.context_state;
      old.timestamps = std::move(h.timestamps);
    }
  }

  // Keep at most `beam` highest-scoring hypotheses.
  void TopK(int32_t beam) {
    if (static_cast<int32_t>(map_.size()) <= beam) return;
    std::vector<std::pair<float, const std::vector<int32_t> *>> scored;
    scored.reserve(map_.size());
    for (const auto &kv : map_) {
      scored.emplace_back(kv.second.TotScore(), &kv.first);
    }
    std::nth_element(scored.begin(), scored.begin() + beam, scored.end(),
                     [](const auto &a, const auto &b) {
                       return a.first > b.first;
                     });
    std::map<std::vector<int32_t>, Hypothesis> pruned;
    for (int32_t i = 0; i != beam; ++i) {
      auto k = *scored[i].second;
      pruned.emplace(k, std::move(map_[k]));
    }
    map_ = std::move(pruned);
  }

  const Hypothesis *GetMostProbable() const {
    const Hypothesis *best = nullptr;
    float best_score = kNegInf;
    for (const auto &kv : map_) {
      float s = kv.second.TotScore();
      if (s > best_score) {
        best_score = s;
        best = &kv.second;
      }
    }
    return best;
  }

  std::map<std::vector<int32_t>, Hypothesis> &Items() { return map_; }
  const std::map<std::vector<int32_t>, Hypothesis> &Items() const {
    return map_;
  }

 private:
  std::map<std::vector<int32_t>, Hypothesis> map_;
};

// One frame of icefall's `_step_worker`. `topk_log_probs` and `topk_indexes`
// are the per-frame top-K (= beam) acoustic candidates, shared across all
// hypotheses in `B`.
HypothesisList StepWorker(const std::vector<float> &topk_log_probs,
                          const std::vector<int32_t> &topk_indexes,
                          const HypothesisList &B, int32_t beam,
                          int32_t blank_id, const ContextGraphPtr &cg,
                          int32_t frame_idx) {
  HypothesisList next;
  for (const auto &kv : B.Items()) {
    const Hypothesis &hyp = kv.second;
    const float hyp_log_prob = hyp.LogProb();

    for (size_t k = 0; k != topk_indexes.size(); ++k) {
      const int32_t new_token = topk_indexes[k];
      const float log_p = topk_log_probs[k];
      bool update_prefix = false;
      Hypothesis nh = hyp;  // clone

      if (new_token == blank_id) {
        nh.log_prob_non_blank = kNegInf;
        nh.log_prob_blank = hyp_log_prob + log_p;
        next.Add(std::move(nh));
        continue;
      }

      if (!hyp.ys.empty() && hyp.ys.back() == new_token) {
        // Branch 1: same prefix, contributes to non-blank stream only.
        nh.log_prob_non_blank = hyp.log_prob_non_blank + log_p;
        nh.log_prob_blank = kNegInf;
        next.Add(std::move(nh));

        // Branch 2: new prefix [..., c], coming from the blank stream.
        Hypothesis nh2 = hyp;  // re-clone
        nh2.ys.push_back(new_token);
        nh2.log_prob_non_blank = hyp.log_prob_blank + log_p;
        nh2.log_prob_blank = kNegInf;
        nh2.timestamps.push_back(frame_idx);
        update_prefix = true;
        if (update_prefix && cg) {
          auto cr = cg->ForwardOneStep(hyp.context_state, new_token);
          nh2.lm_score = hyp.lm_score + cr.first;
          nh2.context_state = cr.second;
        }
        next.Add(std::move(nh2));
        continue;
      }

      // Different token: extend prefix from full log_prob.
      nh.ys.push_back(new_token);
      nh.log_prob_non_blank = hyp_log_prob + log_p;
      nh.log_prob_blank = kNegInf;
      nh.timestamps.push_back(frame_idx);
      update_prefix = true;
      if (update_prefix && cg) {
        auto cr = cg->ForwardOneStep(hyp.context_state, new_token);
        nh.lm_score = hyp.lm_score + cr.first;
        nh.context_state = cr.second;
      }
      next.Add(std::move(nh));
    }
  }

  next.TopK(beam);
  return next;
}

}  // namespace

OfflineCtcPrefixBeamSearchShallowFusionDecoder::
    OfflineCtcPrefixBeamSearchShallowFusionDecoder(int32_t beam,
                                                   int32_t blank_id /*= 0*/)
    : beam_(beam), blank_id_(blank_id) {
  SHERPA_CHECK_GT(beam_, 0);
}

std::vector<OfflineCtcDecoderResult>
OfflineCtcPrefixBeamSearchShallowFusionDecoder::Decode(
    torch::Tensor log_prob, torch::Tensor log_prob_len,
    int32_t subsampling_factor /*= 1*/) {
  std::vector<ContextGraphPtr> empty(log_prob.size(0));
  return Decode(log_prob, log_prob_len, empty, subsampling_factor);
}

std::vector<OfflineCtcDecoderResult>
OfflineCtcPrefixBeamSearchShallowFusionDecoder::Decode(
    torch::Tensor log_prob, torch::Tensor log_prob_len,
    const std::vector<ContextGraphPtr> &context_graphs,
    int32_t subsampling_factor /*= 1*/) {
  InferenceMode no_grad;

  TORCH_CHECK(log_prob.dim() == 3, "log_prob.dim() = ", log_prob.dim());
  TORCH_CHECK(log_prob_len.dim() == 1, "log_prob_len.dim() = ",
              log_prob_len.dim());

  // Algorithm is sequential / map-keyed; CPU-only by design.
  log_prob = log_prob.to(torch::kCPU).to(torch::kFloat).contiguous();
  log_prob_len = log_prob_len.to(torch::kCPU).to(torch::kLong).contiguous();

  const int32_t N = log_prob.size(0);
  const int32_t T = log_prob.size(1);
  const int32_t V = log_prob.size(2);

  TORCH_CHECK(static_cast<int32_t>(context_graphs.size()) == N,
              "context_graphs size ", context_graphs.size(), " vs batch ", N);
  TORCH_CHECK(blank_id_ >= 0 && blank_id_ < V,
              "blank_id ", blank_id_, " out of vocab ", V);

  // Per-frame top-K = beam (icefall: `ctc_output.topk(beam)`).
  const int32_t k = std::min(beam_, V);

  // Compute top-K once for all (N, T) frames using torch's topk for speed.
  torch::Tensor topk_vals_t, topk_idx_t;
  std::tie(topk_vals_t, topk_idx_t) =
      log_prob.topk(/*k*/ k, /*dim*/ -1, /*largest*/ true, /*sorted*/ true);
  topk_vals_t = topk_vals_t.contiguous();
  topk_idx_t = topk_idx_t.to(torch::kInt32).contiguous();

  auto vals_acc = topk_vals_t.accessor<float, 3>();
  auto idx_acc = topk_idx_t.accessor<int32_t, 3>();
  auto len_acc = log_prob_len.accessor<int64_t, 1>();

  std::vector<OfflineCtcDecoderResult> results(N);

  for (int32_t n = 0; n != N; ++n) {
    int32_t T_n = static_cast<int32_t>(len_acc[n]);
    if (T_n > T) T_n = T;

    const ContextGraphPtr &cg = context_graphs[n];
    const ContextState *root = cg ? cg->Root() : nullptr;

    // Initial hypothesis: empty ys, log_prob_blank = 0, log_prob_non_blank = -inf.
    HypothesisList B;
    {
      Hypothesis init;
      init.log_prob_blank = 0.0f;
      init.log_prob_non_blank = kNegInf;
      init.context_state = root;
      B.Add(std::move(init));
    }

    std::vector<float> topk_lp(k);
    std::vector<int32_t> topk_idx(k);

    for (int32_t t = 0; t != T_n; ++t) {
      for (int32_t i = 0; i != k; ++i) {
        topk_lp[i] = vals_acc[n][t][i];
        topk_idx[i] = idx_acc[n][t][i];
      }
      B = StepWorker(topk_lp, topk_idx, B, beam_, blank_id_, cg, t);
    }

    // Finalize context: discount any partial match (icefall does this in
    // the outer function, after the per-frame loop).
    if (cg) {
      for (auto &kv : B.Items()) {
        auto fr = cg->Finalize(kv.second.context_state);
        kv.second.lm_score += fr.first;
        kv.second.context_state = fr.second;
      }
    }

    const Hypothesis *best = B.GetMostProbable();
    if (best != nullptr) {
      results[n].tokens = best->ys;
      results[n].timestamps = best->timestamps;
    }
  }

  (void)subsampling_factor;  // timestamps remain in encoder frames.
  return results;
}

}  // namespace sherpa
