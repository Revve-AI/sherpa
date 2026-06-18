// sherpa/csrc/offline-ctc-prefix-beam-search-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa/csrc/offline-ctc-prefix-beam-search-decoder.h"

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

struct PrefixState {
  // Two probability streams (Hannun 2014):
  //   log_pb  = log P(prefix ending in blank at this frame)
  //   log_pnb = log P(prefix ending in non-blank at this frame)
  float log_pb = kNegInf;
  float log_pnb = kNegInf;

  // Accumulated contextual-bias bonus from ContextGraph::ForwardOneStep.
  float context_score = 0.0f;
  const ContextState *context_state = nullptr;

  // Last non-blank token (needed for repeat handling). -1 if none.
  int32_t last_token = -1;

  // Frame indices (post-subsampling) where each token in `tokens` was first
  // emitted. Same length as `tokens`.
  std::vector<int32_t> timestamps;

  // Score used for beam pruning and final ranking.
  float Score() const {
    return LogAdd(log_pb, log_pnb) + context_score;
  }
};

// Map from token sequence -> PrefixState. We use std::map so the underlying
// key (the prefix tokens) is a stable identity in memory; we don't iterate in
// order beyond pruning by score.
using PrefixMap = std::map<std::vector<int32_t>, PrefixState>;

// Merge contribution into prefix[tokens].
inline PrefixState &GetOrInit(PrefixMap *prefix_map,
                              const std::vector<int32_t> &tokens) {
  auto it = prefix_map->find(tokens);
  if (it != prefix_map->end()) return it->second;
  return prefix_map->emplace(tokens, PrefixState{}).first->second;
}

}  // namespace

OfflineCtcPrefixBeamSearchDecoder::OfflineCtcPrefixBeamSearchDecoder(
    int32_t num_active_paths, int32_t top_k, int32_t blank_id /*= 0*/)
    : num_active_paths_(num_active_paths),
      top_k_(top_k),
      blank_id_(blank_id) {
  SHERPA_CHECK_GT(num_active_paths_, 0);
  SHERPA_CHECK_GE(top_k_, 0);
}

std::vector<OfflineCtcDecoderResult> OfflineCtcPrefixBeamSearchDecoder::Decode(
    torch::Tensor log_prob, torch::Tensor log_prob_len,
    int32_t subsampling_factor /*= 1*/) {
  std::vector<ContextGraphPtr> empty(log_prob.size(0));
  return Decode(log_prob, log_prob_len, empty, subsampling_factor);
}

std::vector<OfflineCtcDecoderResult> OfflineCtcPrefixBeamSearchDecoder::Decode(
    torch::Tensor log_prob, torch::Tensor log_prob_len,
    const std::vector<ContextGraphPtr> &context_graphs,
    int32_t subsampling_factor /*= 1*/) {
  InferenceMode no_grad;

  TORCH_CHECK(log_prob.dim() == 3, "log_prob.dim() = ", log_prob.dim());
  TORCH_CHECK(log_prob_len.dim() == 1, "log_prob_len.dim() = ",
              log_prob_len.dim());

  // CTC prefix beam search is iterative over time; it lives on CPU.
  log_prob = log_prob.to(torch::kCPU).to(torch::kFloat).contiguous();
  log_prob_len = log_prob_len.to(torch::kCPU).to(torch::kLong).contiguous();

  int32_t N = log_prob.size(0);
  int32_t T = log_prob.size(1);
  int32_t V = log_prob.size(2);

  TORCH_CHECK(static_cast<int32_t>(context_graphs.size()) == N,
              "context_graphs size ", context_graphs.size(), " vs batch ", N);
  TORCH_CHECK(blank_id_ >= 0 && blank_id_ < V,
              "blank_id ", blank_id_, " out of vocab ", V);

  // Effective top-k (clamp to vocab size).
  int32_t effective_top_k = top_k_ > 0 ? std::min(top_k_, V) : V;

  std::vector<OfflineCtcDecoderResult> results(N);

  auto lp_acc = log_prob.accessor<float, 3>();
  auto len_acc = log_prob_len.accessor<int64_t, 1>();

  for (int32_t n = 0; n != N; ++n) {
    int32_t T_n = static_cast<int32_t>(len_acc[n]);
    if (T_n > T) T_n = T;

    const ContextGraphPtr &cg = context_graphs[n];
    const ContextState *root = cg ? cg->Root() : nullptr;

    // Initial beam: single empty prefix with log_pb = 0 (probability 1 of
    // being in "blank" state before consuming any frame).
    PrefixMap beam;
    {
      PrefixState init;
      init.log_pb = 0.0f;
      init.log_pnb = kNegInf;
      init.context_state = root;
      beam.emplace(std::vector<int32_t>{}, init);
    }

    // Per-frame top-k token indices (reused buffer).
    std::vector<std::pair<float, int32_t>> token_scores(V);

    for (int32_t t = 0; t != T_n; ++t) {
      const float *lp = &lp_acc[n][t][0];

      // Select top-K tokens by log-prob (always include blank).
      token_scores.resize(V);
      for (int32_t v = 0; v != V; ++v) token_scores[v] = {lp[v], v};

      std::vector<std::pair<float, int32_t>> candidates;
      if (effective_top_k == V) {
        candidates = token_scores;
      } else {
        std::partial_sort(token_scores.begin(),
                          token_scores.begin() + effective_top_k,
                          token_scores.end(),
                          [](const std::pair<float, int32_t> &a,
                             const std::pair<float, int32_t> &b) {
                            return a.first > b.first;
                          });
        candidates.assign(token_scores.begin(),
                          token_scores.begin() + effective_top_k);
        // Ensure blank is among candidates.
        bool has_blank = false;
        for (const auto &c : candidates) {
          if (c.second == blank_id_) {
            has_blank = true;
            break;
          }
        }
        if (!has_blank) candidates.emplace_back(lp[blank_id_], blank_id_);
      }

      PrefixMap next_beam;

      for (const auto &kv : beam) {
        const std::vector<int32_t> &tokens = kv.first;
        const PrefixState &p = kv.second;
        float p_b = p.log_pb;
        float p_nb = p.log_pnb;

        for (const auto &cand : candidates) {
          int32_t c = cand.second;
          float log_pc = cand.first;

          if (c == blank_id_) {
            // Stay on the same prefix; only updates blank stream.
            PrefixState &nxt = GetOrInit(&next_beam, tokens);
            // First time we touch nxt: copy bias state from parent.
            if (nxt.context_state == nullptr && nxt.last_token == -1 &&
                nxt.log_pb == kNegInf && nxt.log_pnb == kNegInf) {
              nxt.context_state = p.context_state;
              nxt.context_score = p.context_score;
              nxt.last_token = p.last_token;
              nxt.timestamps = p.timestamps;
            }
            nxt.log_pb = LogAdd(nxt.log_pb, LogAdd(p_b, p_nb) + log_pc);
            continue;
          }

          if (!tokens.empty() && c == p.last_token) {
            // Repeat (no new symbol): contributes only via the non-blank
            // stream of the same prefix.
            PrefixState &nxt = GetOrInit(&next_beam, tokens);
            if (nxt.context_state == nullptr && nxt.last_token == -1 &&
                nxt.log_pb == kNegInf && nxt.log_pnb == kNegInf) {
              nxt.context_state = p.context_state;
              nxt.context_score = p.context_score;
              nxt.last_token = p.last_token;
              nxt.timestamps = p.timestamps;
            }
            nxt.log_pnb = LogAdd(nxt.log_pnb, p_nb + log_pc);

            // Extend a NEW prefix [..., c] via the blank stream (so we emit
            // a second occurrence of c after a blank).
            std::vector<int32_t> new_tokens = tokens;
            new_tokens.push_back(c);
            PrefixState &nxt2 = GetOrInit(&next_beam, new_tokens);
            if (nxt2.context_state == nullptr && nxt2.last_token == -1 &&
                nxt2.log_pb == kNegInf && nxt2.log_pnb == kNegInf) {
              float boost = 0.0f;
              const ContextState *new_state = p.context_state;
              if (cg) {
                auto cr = cg->ForwardOneStep(p.context_state, c);
                boost = cr.first;
                new_state = cr.second;
              }
              nxt2.context_state = new_state;
              nxt2.context_score = p.context_score + boost;
              nxt2.last_token = c;
              nxt2.timestamps = p.timestamps;
              nxt2.timestamps.push_back(t);
            }
            nxt2.log_pnb = LogAdd(nxt2.log_pnb, p_b + log_pc);
            continue;
          }

          // New non-repeat token: extend prefix.
          std::vector<int32_t> new_tokens = tokens;
          new_tokens.push_back(c);
          PrefixState &nxt = GetOrInit(&next_beam, new_tokens);
          if (nxt.context_state == nullptr && nxt.last_token == -1 &&
              nxt.log_pb == kNegInf && nxt.log_pnb == kNegInf) {
            float boost = 0.0f;
            const ContextState *new_state = p.context_state;
            if (cg) {
              auto cr = cg->ForwardOneStep(p.context_state, c);
              boost = cr.first;
              new_state = cr.second;
            }
            nxt.context_state = new_state;
            nxt.context_score = p.context_score + boost;
            nxt.last_token = c;
            nxt.timestamps = p.timestamps;
            nxt.timestamps.push_back(t);
          }
          nxt.log_pnb = LogAdd(nxt.log_pnb, LogAdd(p_b, p_nb) + log_pc);
        }  // for each candidate token
      }    // for each prefix in beam

      // Prune to top num_active_paths_ by Score().
      if (static_cast<int32_t>(next_beam.size()) > num_active_paths_) {
        std::vector<std::pair<float, const std::vector<int32_t> *>> scored;
        scored.reserve(next_beam.size());
        for (const auto &kv : next_beam) {
          scored.emplace_back(kv.second.Score(), &kv.first);
        }
        std::nth_element(
            scored.begin(), scored.begin() + num_active_paths_, scored.end(),
            [](const std::pair<float, const std::vector<int32_t> *> &a,
               const std::pair<float, const std::vector<int32_t> *> &b) {
              return a.first > b.first;
            });
        PrefixMap pruned;
        for (int32_t i = 0; i != num_active_paths_; ++i) {
          const auto &k = *scored[i].second;
          pruned.emplace(k, std::move(next_beam[k]));
        }
        beam = std::move(pruned);
      } else {
        beam = std::move(next_beam);
      }
    }  // for t

    // Pick the best prefix; apply Finalize() to discount any partial match.
    const std::vector<int32_t> *best_tokens = nullptr;
    const PrefixState *best_state = nullptr;
    float best_score = kNegInf;
    for (const auto &kv : beam) {
      float s = kv.second.Score();
      if (cg) {
        auto fr = cg->Finalize(kv.second.context_state);
        s += fr.first;
      }
      if (s > best_score) {
        best_score = s;
        best_tokens = &kv.first;
        best_state = &kv.second;
      }
    }

    OfflineCtcDecoderResult &r = results[n];
    if (best_tokens != nullptr) {
      r.tokens = *best_tokens;
      r.timestamps = best_state->timestamps;
    }
  }  // for n

  (void)subsampling_factor;  // timestamps are in encoder frames; conversion to
                             // seconds is done by the recognizer using
                             // subsampling_factor.
  return results;
}

}  // namespace sherpa
