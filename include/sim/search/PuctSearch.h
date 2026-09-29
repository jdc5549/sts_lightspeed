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
#include "sim/search/GameLegality.h"
#include "sim/search/PyLegality.h"

namespace sts::search {

    // Exactly BattleContext::clone_with_fresh_rng as the pybind binding does it.
    BattleContext cloneWithFreshRng(const BattleContext &bc, std::uint64_t seed, bool reshuffleDrawPile);

    // ---- pure pieces, exported so the unit tests can hammer them against the Python expressions ----
    float wAdd(float w, double v);                                   // node.W[a] += v
    float noiseMix(float p, double eps, double noise);               // node.P[a] = (1.0-eps)*node.P[a] + eps*noise
    // _puct_select's scalar loop over ascending legal indices; requires mask.any().
    int puctSelectRaw(const std::int32_t *N, const float *W, const float *P, const bool *mask, double cPuct, int mutation = 0);
    // S.2: PUCT selection under virtual loss. Effective counts are N' = N + vl (int32) and effective totals
    // W' = double(W) + double(vl) * vLoss (DOUBLE precision, computed per edge from the real float32 W); the
    // rest is puctSelectRaw's expression on (N', W') (sum of N' in the sqrt, strict > first max). The caller
    // dispatches here ONLY when the node has virtual loss in flight; with all-zero vl the real-N/W path runs.
    int puctSelectVL(const std::int32_t *N, const std::int32_t *vl, const float *W, const float *P,
                     const bool *mask, double cPuct, double vLoss);

    enum class TerminalMode { Outcome = 0, HpFraction = 1 };

    struct LeafResult {
        std::array<float, PY_ACTION_SPACE> P;
        double value = 0.0;
    };
    // Called with the GIL held on an nn-cache MISS. Runs the net + _expand's post-processing.
    using LeafEval = std::function<LeafResult(const Features &, const PyMask76 &)>;

    // S.1 (lockstep across the K trees): ONE call evaluates every distinct pending leaf of a step. Rows are
    // the distinct nn keys in ascending-tree-index first-occurrence order; results[i] answers row i.
    using BatchLeafEval = std::function<std::vector<LeafResult>(const std::vector<const Features *> &,
                                                                const std::vector<const PyMask76 *> &)>;

    struct TreeResult {
        std::array<std::int32_t, PY_ACTION_SPACE> rawN{};
        double rootValue = 0.0;
    };

    struct SearchCounters {
        long long nnHits = 0, nnMisses = 0, extractHits = 0, extractMisses = 0, leafEvals = 0,
                  sims = 0, trees = 0, nodes = 0, actionsApplied = 0;
        // S.1 lockstep: steps run, batch callbacks issued, pending leaves submitted (incl. duplicates
        // that share a row), distinct rows evaluated, and the largest batch.
        long long lockstepSteps = 0, batchCalls = 0, leavesSubmitted = 0, distinctRows = 0, maxBatch = 0;
        // S.2 virtual loss: descents that ended pending and were charged (collisions included), collisions
        // (a descent that reached a node already pending in the same step and tree), the largest number of
        // pending descents in one step, the peak node count of any one tree (memory per resident tree =
        // peakTreeNodes * PuctSearch::nodeBytes()), and the largest virtual-loss charge still in flight
        // AFTER any step's completion (must be 0: every charge is reversed).
        long long vlDescents = 0, vlCollisions = 0, vlPendingMax = 0, peakTreeNodes = 0, vlInflightAfterStepMax = 0;
    };

    // TEST/diagnostic record of one pending leaf of a lockstep step (see setRecordLockstepLog).
    struct LockstepLogEntry {
        int tree;              // tree index
        std::string nnKey;     // the state key that keyed the nn cache
        int row;               // index of the row that answers it in the batch (first-occurrence order)
        std::string sig;       // the row's section keys, concatenated in FeatSection order
    };

    class PuctSearch {
    public:
        explicit PuctSearch(const FeaturizerConfig &cfg) : cfg_(cfg) {}

        // Per-combat caches (MCTSMicroAgent.set_bc clears both Python caches).
        void clear();
        // S.3: which 76-slot mask the tree uses at every node (root included): the Python-enumerator emulation
        // (default) or the simulator's own legality (legalMask76Game). Switching clears the per-combat caches
        // (their entries carry the mask of the mode that produced them).
        void setLegalityGame(bool game) { if (game != legalityGame_) { legalityGame_ = game; clear(); } }
        bool legalityGame() const { return legalityGame_; }
        PyMask76 legalMask(const BattleContext &bc) const { return legalityGame_ ? legalMask76Game(bc) : legalMask76(bc); }
        const FeaturizerConfig &config() const { return cfg_; }

        // ONE determinization: bcK = clone_with_fresh_rng(realBc, seed, reshuffle), then nSims sims.
        // rootNoise: the Dirichlet draw for the root's legal actions (ascending index) or nullptr.
        TreeResult runTree(const BattleContext &realBc, std::uint64_t seed, bool reshuffle, int nSims,
                           double cPuct, const std::vector<double> *rootNoise, double noiseEps,
                           const LeafEval &leafEval, TerminalMode terminal);

        // S.1: the K determinizations advance TOGETHER. Each step every live tree runs ONE simulation down
        // to a finish that needs no net call (terminal / incompatible / all-False mask / nn hit) or to a
        // leaf that needs one; the pending leaves are evaluated in ONE batchEval call (rows = distinct nn
        // keys in ascending-tree-index first-occurrence order, no padding), then each pending tree
        // completes its expand + backup in ascending tree index. Same PUCT, numerics, per-tree pools,
        // shared caches and terminal values as runTree; root noise (values passed in, one nullable pointer
        // per tree) is applied to tree k right after ITS sim 0 completes. If every leaf were evaluated at
        // batch 1 the result is identical to running the trees one after another, for any K.
        //
        // S.2 VIRTUAL LOSS (vlBatch = B > 1): step 0 is unchanged (ONE descent per tree: the root expansion,
        // then that tree's root noise); from step 1 each tree does up to min(B, remaining sims) descents per
        // step, trees ascending, descents in order. The virtual loss lives in SEPARATE per-node arrays
        // (Node::vlN), never in the real N/W (a float32 W += v then -= v is not bit-reversible): selection
        // at a node with vlSum > 0 reads N' = N + vlN, W' = W + vlN * vlValue (puctSelectVL); at a node with
        // vlSum == 0 it runs today's exact code on real N/W, which is what makes B == 1 byte-identical to S.1.
        // A descent that ends PENDING (needs a leaf eval) charges vlN along its path; one that finishes
        // without a net call backs up its real value at once, uncharged. A descent that reaches a node already
        // pending in this step (same tree) is a COLLISION: it stops there, is registered on the same node/row,
        // counts as a sim, and (charged like any pending descent) backs up the node's value when the step
        // completes; the node is expanded once. After the batched callback each pending descent, in
        // (tree, order), releases its charge, expands its node (first time) and backs up with wAdd. After every
        // step all vlN are zero. rootValue = sum of real backed-up values / nSims in completion order.
        std::vector<TreeResult> runTreesLockstep(const BattleContext &realBc, const std::vector<std::uint64_t> &seeds,
                                                 bool reshuffle, int nSims, double cPuct,
                                                 const std::vector<const std::vector<double> *> &rootNoises,
                                                 double noiseEps, const BatchLeafEval &batchEval,
                                                 TerminalMode terminal, int vlBatch = 1, double vlValue = -1.0);
        const std::vector<std::array<float, PY_ACTION_SPACE>> &lastRootWs() const { return lastRootWs_; }
        void setRecordLockstepLog(bool on) { recordLog_ = on; }
        // TEST: scan every node of every tree after a lockstep search and throw if any vlN is nonzero.
        void setDebugChecks(bool on) { debugChecks_ = on; }
        static std::size_t nodeBytes();
        const std::vector<std::vector<LockstepLogEntry>> &lockstepLog() const { return lockstepLog_; }
        void clearLockstepLog() { lockstepLog_.clear(); }

        // Diagnostics/tests: the last tree's root W (float32) -- W is not visible in the visit counts.
        const std::array<float, PY_ACTION_SPACE> &lastRootW() const { return lastRootW_; }
        const SearchCounters &counters() const { return counters_; }
        void resetCounters() { counters_ = SearchCounters(); }
        std::size_t nnCacheSize() const { return nn_.size(); }
        std::size_t extractCacheSize() const { return extract_.size(); }

        // TEST ONLY (the C.4e mutation checks): 0 = exact. 1 = PUCT tie-break by LAST max; 2 = the
        // root's noised P is written back into the nn cache; 3 = the root is counted as visited on sim
        // 0; 4 = W accumulated in double; 5 (lockstep only) = the batch rows are handed to the callback in
        // REVERSE order (results mapped back, so only the composition moves); 6 (virtual loss) = the charge is
        // never reversed; 7 (virtual loss) = the charge is written into the REAL N/W and reversed there (float32
        // W += v then -= v does not round-trip). Never set outside tests.
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
            // S.2 virtual loss: per-edge count of in-flight descents, their total, and the lockstep step in
            // which this node was last registered as a pending leaf (-1 = never).
            std::array<std::int32_t, PY_ACTION_SPACE> vlN{};
            std::int32_t vlSum = 0;
            std::int32_t pendStep = -1;
            BattleContext bc;
            Node() { child.fill(-1); }
        };

        struct PathEnt { Node *node; int action; };
        // One simulation's descent state: the path so far and (when it stopped at a leaf that needs the
        // net) the leaf node and its bundle.
        struct SimCursor {
            std::vector<PathEnt> path;
            Node *node = nullptr;
            const Bundle *state = nullptr;
            bool collision = false;        // pending on a node already pending in this step (same tree)
        };

        Bundle makeBundle(const BattleContext &bc, bool wantFeatures) const;
        double expand(Node &node, const Bundle &state, const LeafEval *leafEval);
        double simulate(std::deque<Node> &pool, const Bundle &rootState, double cPuct,
                        const LeafEval &leafEval, TerminalMode terminal);
        void backup(std::vector<PathEnt> &path, double v);
        // Descend one simulation. Returns true when it finished (value in v, backup done). With
        // leafEval == nullptr an nn-cache MISS at the leaf returns false (cursor.node/state = the leaf)
        // instead of evaluating it; with a leafEval it never returns false.
        // `vlStep` >= 0 (lockstep only): the current step, enabling collision detection and virtual-loss-aware
        // selection at nodes that carry a charge (`vlValue` = the per-descent value).
        bool descend(std::deque<Node> &pool, const Bundle &rootState, double cPuct, const LeafEval *leafEval,
                     TerminalMode terminal, SimCursor &cur, double &v, int vlStep = -1, double vlValue = -1.0);
        void vlCharge(std::vector<PathEnt> &path, double vlValue);
        void vlRelease(std::vector<PathEnt> &path, double vlValue);
        void applyRootNoise(Node &root, const Bundle &rootState, const std::vector<double> &noise, double eps);
        static double terminalValue(const BattleContext &bc, TerminalMode mode);

        FeaturizerConfig cfg_;
        std::unordered_map<std::string, NNEntry> nn_;
        std::unordered_map<std::uint64_t, Bundle> extract_;
        SearchCounters counters_;
        std::array<float, PY_ACTION_SPACE> lastRootW_{};
        std::vector<std::array<float, PY_ACTION_SPACE>> lastRootWs_;
        std::vector<std::vector<LockstepLogEntry>> lockstepLog_;
        bool recordLog_ = false;
        bool legalityGame_ = false;
        bool debugChecks_ = false;
        long long vlInflight_ = 0;
        int mutation_ = 0;
    };

}

#endif
