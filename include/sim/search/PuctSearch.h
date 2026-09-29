// C++ PUCT tree reproducing src/sts/agent/mcts.py (_run_mcts_raw + _simulate + _expand + _puct_select +
// _backup + _inject_root_noise) BIT FOR BIT (PLAN-perf-cpp-search C.4e, exact stage, decision 3).
// The numerics stay in Python where Python does them today: a leaf-evaluation CALLBACK (net forward +
// _expand's softmax/mask/renorm post-processing) runs on an nn-cache MISS. Everything else -- the
// per-combat caches (nn cache keyed on the CombatState-equivalent stateKey, extract cache keyed on
// bc.state_hash()), the tree, PUCT, backup, root noise, terminal values -- is here.
//
// Storage choice: per-NODE BattleContext (sizeof(BattleContext) == 3408 bytes, so a 300-sim tree is
// ~1.4 MB with the node arrays) instead of replaying the path from a clone every sim as Python does.
// Both are exact because action application is deterministic (C.4d compares RNG streams); the stored
// form does one copy + one action per new node instead of one copy + depth actions per sim.
//
// NUMERIC RULES (numpy 2.4 / NEP 50, measured, see PuctSearch.cpp):
//   W[a] += v            ==  W = f32( W + f32(v) )                      (float32 add of the rounded v)
//   P[a] = (1-eps)*P[a] + eps*noise   ==  f32( f32(f32(1-eps)*P) + f32(eps*noise) ) (eps*noise in double)
#ifndef STS_LIGHTSPEED_PUCTSEARCH_H
#define STS_LIGHTSPEED_PUCTSEARCH_H

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "combat/BattleContext.h"
#include "sim/search/ApplyAction.h"
#include "sim/search/Featurize.h"
#include "sim/search/PyLegality.h"

namespace sts::search {

    // Exactly BattleContext::clone_with_fresh_rng as the pybind binding does it.
    BattleContext cloneWithFreshRng(const BattleContext &bc, std::uint64_t seed, bool reshuffleDrawPile);

    // ---- pure pieces, exported so the unit tests can hammer them against the Python expressions ----
    float wAdd(float w, double v);                                   // node.W[a] += v
    float noiseMix(float p, double eps, double noise);               // node.P[a] = (1.0-eps)*node.P[a] + eps*noise
    // _puct_select's scalar loop over ascending legal indices; requires mask.any().
    int puctSelectRaw(const std::int32_t *N, const float *W, const float *P, const bool *mask, double cPuct, int mutation = 0);

    enum class TerminalMode { Outcome = 0, HpFraction = 1 };

    struct LeafResult {
        std::array<float, PY_ACTION_SPACE> P;
        double value = 0.0;
    };
    // Called with the GIL held on an nn-cache MISS. Runs the net + _expand's post-processing.
    using LeafEval = std::function<LeafResult(const Features &, const PyMask76 &)>;

    struct TreeResult {
        std::array<std::int32_t, PY_ACTION_SPACE> rawN{};
        double rootValue = 0.0;
    };

    struct SearchCounters {
        long long nnHits = 0, nnMisses = 0, extractHits = 0, extractMisses = 0, leafEvals = 0,
                  sims = 0, trees = 0, nodes = 0, actionsApplied = 0;
    };

    class PuctSearch {
    public:
        explicit PuctSearch(const FeaturizerConfig &cfg) : cfg_(cfg) {}

        // Per-combat caches (MCTSMicroAgent.set_bc clears both Python caches).
        void clear();
        const FeaturizerConfig &config() const { return cfg_; }

        // ONE determinization: bcK = clone_with_fresh_rng(realBc, seed, reshuffle), then nSims sims.
        // rootNoise: the Dirichlet draw for the root's legal actions (ascending index) or nullptr.
        TreeResult runTree(const BattleContext &realBc, std::uint64_t seed, bool reshuffle, int nSims,
                           double cPuct, const std::vector<double> *rootNoise, double noiseEps,
                           const LeafEval &leafEval, TerminalMode terminal);

        // Diagnostics/tests: the last tree's root W (float32) -- W is not visible in the visit counts.
        const std::array<float, PY_ACTION_SPACE> &lastRootW() const { return lastRootW_; }
        const SearchCounters &counters() const { return counters_; }
        void resetCounters() { counters_ = SearchCounters(); }
        std::size_t nnCacheSize() const { return nn_.size(); }
        std::size_t extractCacheSize() const { return extract_.size(); }

        // TEST ONLY (the C.4e mutation checks): 0 = exact. 1 = PUCT tie-break by LAST max; 2 = the
        // root's noised P is written back into the nn cache; 3 = the root is counted as visited on sim
        // 0; 4 = W accumulated in double. Never set outside tests.
        void setMutation(int m) { mutation_ = m; }

    private:
        // Everything extract_combat_state(bc, gc) yields that the tree needs, captured at first sight.
        struct Bundle {
            std::string key;
            PyMask76 mask{};
            bool compat = false;
            ConvCtx ctx;
            std::unique_ptr<Features> feat;   // only when the leaf will need it (compat && nn miss at extraction)
        };
        struct NNEntry {
            PyMask76 mask;
            std::array<float, PY_ACTION_SPACE> P;
            double value;
        };
        struct Node {
            std::array<float, PY_ACTION_SPACE> P{}, W{};
            std::array<std::int32_t, PY_ACTION_SPACE> N{};
            std::array<std::int32_t, PY_ACTION_SPACE> child;
            PyMask76 mask{};
            bool expanded = false;
            const Bundle *state = nullptr;
            BattleContext bc;
            Node() { child.fill(-1); }
        };

        Bundle makeBundle(const BattleContext &bc, bool wantFeatures) const;
        double expand(Node &node, const Bundle &state, const LeafEval &leafEval);
        double simulate(std::deque<Node> &pool, const Bundle &rootState, double cPuct,
                        const LeafEval &leafEval, TerminalMode terminal);
        static double terminalValue(const BattleContext &bc, TerminalMode mode);

        FeaturizerConfig cfg_;
        std::unordered_map<std::string, NNEntry> nn_;
        std::unordered_map<std::uint64_t, Bundle> extract_;
        SearchCounters counters_;
        std::array<float, PY_ACTION_SPACE> lastRootW_{};
        int mutation_ = 0;
    };

}

#endif
