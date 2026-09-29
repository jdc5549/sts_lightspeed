// See include/sim/search/PuctSearch.h. Mirrors src/sts/agent/mcts.py line by line.
// No floating-point contraction anywhere in this file: the Python side never fuses.
#pragma GCC optimize("fp-contract=off")

#include "sim/search/PuctSearch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "sim/search/StateHash.h"
#include "sim/search/StateKey.h"

namespace sts::search {

    BattleContext cloneWithFreshRng(const BattleContext &bc, std::uint64_t seed, bool reshuffleDrawPile) {
        BattleContext copy = bc;
        copy.aiRng         = sts::Random(seed + 0);
        copy.cardRandomRng = sts::Random(seed + 1);
        copy.miscRng       = sts::Random(seed + 2);
        copy.monsterHpRng  = sts::Random(seed + 3);
        copy.potionRng     = sts::Random(seed + 4);
        copy.shuffleRng    = sts::Random(seed + 5);
        if (reshuffleDrawPile) {
            auto &dp = copy.cards.drawPile;
            java::Collections::shuffle(dp.begin(), dp.end(), java::Random(copy.shuffleRng.randomLong()));
        }
        return copy;
    }

    // numpy 2.4 (NEP 50): `node.W[a] += v` with W float32 and v a Python float. The Python float is weak,
    // so it is cast to float32 FIRST and the add is a float32 add (measured: 0 mismatches / 3e5 vs this
    // form; the "add in double then round" form mismatches ~18%).
    float wAdd(float w, double v) {
        const float vf = static_cast<float>(v);
        const float r = w + vf;
        return r;
    }

    // `(1.0 - eps) * node.P[a] + eps * float(noise[i])`, P[a] an np.float32 scalar:
    // f32(1.0-eps) * P in float32; eps*noise is a pure-Python double product, cast to float32 when it meets
    // the float32 scalar; float32 add. (Measured 0 mismatches / 8e5 across eps in {.3,.17,.25,.125}.)
    float noiseMix(float p, double eps, double noise) {
        const float omE = static_cast<float>(1.0 - eps);
        const float t1 = omE * p;
        const float t2 = static_cast<float>(eps * noise);
        const float r = t1 + t2;
        return r;
    }

    static int puctSelectImpl(const std::int32_t *N, const float *W, const float *P, const bool *mask,
                              double cPuct, int mutation) {
        long long sumN = 0;
        for (int i = 0; i < PY_ACTION_SPACE; ++i) sumN += N[i];
        const double sqrtSumN = sumN > 0 ? std::sqrt(static_cast<double>(sumN)) : 0.0;
        int first = -1;
        for (int i = 0; i < PY_ACTION_SPACE; ++i) if (mask[i]) { first = i; break; }
        if (first < 0) throw std::logic_error("puctSelect: no legal action");
        double bestScore = -std::numeric_limits<double>::infinity();
        int bestAction = first;
        for (int a = 0; a < PY_ACTION_SPACE; ++a) {
            if (!mask[a]) continue;
            const int n = N[a];
            const double w = static_cast<double>(W[a]);
            const double p = static_cast<double>(P[a]);
            const double q = w / static_cast<double>(n > 1 ? n : 1);
            const double u = cPuct * p * sqrtSumN / static_cast<double>(1 + n);
            const double puct = q + u;
            if (mutation == 1 ? (puct >= bestScore) : (puct > bestScore)) {
                bestScore = puct;
                bestAction = a;
            }
        }
        return bestAction;
    }

    int puctSelectRaw(const std::int32_t *N, const float *W, const float *P, const bool *mask, double cPuct,
                      int mutation) {
        return puctSelectImpl(N, W, P, mask, cPuct, mutation);
    }

    int puctSelectVL(const std::int32_t *N, const std::int32_t *vl, const float *W, const float *P,
                     const bool *mask, double cPuct, double vLoss) {
        long long sumN = 0;
        for (int i = 0; i < PY_ACTION_SPACE; ++i) sumN += static_cast<long long>(N[i]) + vl[i];   // N'
        const double sqrtSumN = sumN > 0 ? std::sqrt(static_cast<double>(sumN)) : 0.0;
        int first = -1;
        for (int i = 0; i < PY_ACTION_SPACE; ++i) if (mask[i]) { first = i; break; }
        if (first < 0) throw std::logic_error("puctSelectVL: no legal action");
        double bestScore = -std::numeric_limits<double>::infinity();
        int bestAction = first;
        for (int a = 0; a < PY_ACTION_SPACE; ++a) {
            if (!mask[a]) continue;
            const long long n = static_cast<long long>(N[a]) + vl[a];                               // N'
            const double w = static_cast<double>(W[a]) + static_cast<double>(vl[a]) * vLoss;         // W', double
            const double p = static_cast<double>(P[a]);
            const double q = w / static_cast<double>(n > 1 ? n : 1);
            const double u = cPuct * p * sqrtSumN / static_cast<double>(1 + n);
            const double puct = q + u;
            if (puct > bestScore) { bestScore = puct; bestAction = a; }
        }
        return bestAction;
    }

    std::size_t PuctSearch::nodeBytes() { return sizeof(Node); }

    void PuctSearch::vlCharge(std::vector<PathEnt> &path, double vlValue) {
        for (auto &e : path) {
            if (mutation_ == 7) {                       // TEST ONLY: charge written into the REAL N/W
                e.node->N[e.action] += 1;
                e.node->W[e.action] = wAdd(e.node->W[e.action], vlValue);
            } else {
                e.node->vlN[e.action] += 1;
                e.node->vlSum += 1;
            }
            ++vlInflight_;
        }
    }

    void PuctSearch::vlRelease(std::vector<PathEnt> &path, double vlValue) {
        if (mutation_ == 6) return;                     // TEST ONLY: the reversal is skipped
        for (auto &e : path) {
            if (mutation_ == 7) {                       // "reversed" in float32: does not round-trip
                e.node->N[e.action] -= 1;
                e.node->W[e.action] = e.node->W[e.action] - static_cast<float>(vlValue);
            } else {
                e.node->vlN[e.action] -= 1;
                e.node->vlSum -= 1;
            }
            --vlInflight_;
        }
    }

    void PuctSearch::clear() {
        nn_.clear();
        extract_.clear();
    }

    PuctSearch::Bundle PuctSearch::makeBundle(const BattleContext &bc, bool wantFeatures) const {
        Bundle b;
        b.key = stateKey(bc);
        b.mask = legalMask76(bc);
        b.compat = fixedSpaceCompatible(bc);
        b.ctx = makeConvCtx(bc);
        if (wantFeatures && b.compat && nn_.find(b.key) == nn_.end())
            b.feat = std::make_unique<Features>(featurize(bc, cfg_));
        return b;
    }

    // _expand(node, state, net, nn_cache). `leafEval` may be null only when the nn cache is known to hit.
    double PuctSearch::expand(Node &node, const Bundle &state, const LeafEval *leafEval) {
        auto it = nn_.find(state.key);
        if (it != nn_.end()) {
            ++counters_.nnHits;
            node.mask = it->second.mask;
            node.P = it->second.P;            // a COPY: root noise mutates node.P in place
            node.expanded = true;
            return it->second.value;
        }
        ++counters_.nnMisses;
        if (!state.feat) throw std::logic_error("PuctSearch: nn miss on a bundle extracted without features");
        if (leafEval == nullptr) throw std::logic_error("PuctSearch: nn miss with no leaf evaluator");
        ++counters_.leafEvals;
        const LeafResult r = (*leafEval)(*state.feat, state.mask);
        node.P = r.P;
        node.mask = state.mask;
        node.expanded = true;
        nn_.emplace(state.key, NNEntry{state.mask, r.P, r.value});   // its own copy of P
        return r.value;
    }

    double PuctSearch::terminalValue(const BattleContext &bc, TerminalMode mode) {
        if (mode == TerminalMode::Outcome)
            return bc.outcome == Outcome::PLAYER_VICTORY ? 1.0 : -1.0;
        // HPFractionRewardFunction.compute_terminal_reward_from_bc (forced_end False)
        if (bc.outcome != Outcome::PLAYER_VICTORY) return -1.0;
        const int maxHp = bc.player.maxHp;
        if (maxHp <= 0) return 0.0;
        return static_cast<double>(bc.player.curHp) / static_cast<double>(maxHp);   // int/int true division
    }


    void PuctSearch::backup(std::vector<PathEnt> &path, double v) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {   // reversed(path)
            it->node->N[it->action] += 1;
            if (mutation_ == 4)
                it->node->W[it->action] = static_cast<float>(static_cast<double>(it->node->W[it->action]) + v);
            else
                it->node->W[it->action] = wAdd(it->node->W[it->action], v);
        }
    }

    // _simulate(...). Nodes are pool indices; `pool[0]` is the root.
    double PuctSearch::simulate(std::deque<Node> &pool, const Bundle &rootState, double cPuct,
                                const LeafEval &leafEval, TerminalMode terminal) {
        SimCursor cur;
        double v = 0.0;
        descend(pool, rootState, cPuct, &leafEval, terminal, cur, v);
        return v;
    }

    bool PuctSearch::descend(std::deque<Node> &pool, const Bundle &rootState, double cPuct,
                             const LeafEval *leafEval, TerminalMode terminal, SimCursor &cur, double &vOut,
                             int vlStep, double vlValue) {
        std::vector<PathEnt> &path = cur.path;
        Node *node = &pool[0];
        const Bundle *state = &rootState;

        while (true) {
            if (node->bc.outcome != Outcome::UNDECIDED) {          // terminal check
                const double v = terminalValue(node->bc, terminal);
                backup(path, v);
                vOut = v;
                return true;
            }
            if (!node->expanded) {                                  // leaf: expand
                if (!state->compat) { backup(path, 0.0); vOut = 0.0; return true; }
                if (leafEval == nullptr && vlStep >= 0 && node->pendStep == vlStep) {
                    cur.node = node;                                // COLLISION: already pending in this step
                    cur.state = state;
                    cur.collision = true;
                    return false;
                }
                if (leafEval == nullptr && nn_.find(state->key) == nn_.end()) {
                    cur.node = node;                                // pending: the caller evaluates it
                    cur.state = state;
                    return false;
                }
                const double v = expand(*node, *state, leafEval);
                if (mutation_ == 3 && path.empty())
                    path.push_back({node, puctSelectImpl(node->N.data(), node->W.data(), node->P.data(),
                                                         node->mask.data(), cPuct, 0)});
                backup(path, v);
                vOut = v;
                return true;
            }
            const PyMask76 &mask = node->mask;                      // selection: nn_cache path
            bool any = false;
            for (bool m : mask) any = any || m;
            if (!any) { backup(path, 0.0); vOut = 0.0; return true; }

            // vlSum == 0 (always so on the runTree path and whenever nothing is in flight here): today's exact
            // code on real N/W. Otherwise the virtual-loss-aware selection on N' / W'.
            const int a = node->vlSum == 0
                ? puctSelectImpl(node->N.data(), node->W.data(), node->P.data(), mask.data(), cPuct, mutation_)
                : puctSelectVL(node->N.data(), node->vlN.data(), node->W.data(), node->P.data(), mask.data(),
                               cPuct, vlValue);
            path.push_back({node, a});

            if (node->child[a] < 0) {
                pool.emplace_back();
                Node &child = pool.back();
                child.bc = node->bc;                                // clone parent, apply the action
                applyActionIndex(child.bc, a, state->ctx);
                ++counters_.actionsApplied;
                ++counters_.nodes;
                node->child[a] = static_cast<std::int32_t>(pool.size() - 1);
            }
            node = &pool[static_cast<std::size_t>(node->child[a])];

            if (node->bc.outcome == Outcome::UNDECIDED) {
                if (node->state == nullptr) {
                    const std::uint64_t h = pyStateHash(node->bc);
                    auto it = extract_.find(h);
                    if (it != extract_.end()) {
                        ++counters_.extractHits;
                        node->state = &it->second;
                    } else {
                        ++counters_.extractMisses;
                        auto ins = extract_.emplace(h, makeBundle(node->bc, true));
                        node->state = &ins.first->second;
                    }
                }
                state = node->state;
            }
        }
    }

    // Noise after sim 0 expands the root (always expanded: the caller guarantees the root is fixed-space
    // compatible and not over, so sim 0 reaches _expand at the root).
    void PuctSearch::applyRootNoise(Node &root, const Bundle &rootState, const std::vector<double> &noise,
                                    double eps) {
        std::size_t i = 0;
        for (int a = 0; a < PY_ACTION_SPACE; ++a) {
            if (!rootState.mask[a]) continue;
            if (i >= noise.size()) throw std::invalid_argument("root noise shorter than legal count");
            root.P[a] = noiseMix(root.P[a], eps, noise[i]);
            ++i;
        }
        if (i != noise.size()) throw std::invalid_argument("root noise length != legal count");
        if (mutation_ == 2) nn_[rootState.key].P = root.P;
    }

    TreeResult PuctSearch::runTree(const BattleContext &realBc, std::uint64_t seed, bool reshuffle, int nSims,
                                   double cPuct, const std::vector<double> *rootNoise, double noiseEps,
                                   const LeafEval &leafEval, TerminalMode terminal) {
        ++counters_.trees;
        // The root's key / features / mask / ConvCtx come from the REAL bc (Python's root_state is extracted
        // from it), and the root is NOT put in the extract cache.
        Bundle rootState = makeBundle(realBc, true);
        std::deque<Node> pool;      // deque: node addresses are stable
        pool.emplace_back();
        pool[0].bc = cloneWithFreshRng(realBc, seed, reshuffle);
        ++counters_.nodes;

        double totalValue = 0.0;
        for (int simIdx = 0; simIdx < nSims; ++simIdx) {
            ++counters_.sims;
            const double v = simulate(pool, rootState, cPuct, leafEval, terminal);
            totalValue += v;
            if (rootNoise != nullptr && simIdx == 0 && pool[0].expanded)
                applyRootNoise(pool[0], rootState, *rootNoise, noiseEps);
        }
        TreeResult res;
        res.rawN = pool[0].N;
        lastRootW_ = pool[0].W;
        res.rootValue = nSims > 0 ? totalValue / static_cast<double>(nSims) : 0.0;
        return res;
    }

    std::vector<TreeResult> PuctSearch::runTreesLockstep(
            const BattleContext &realBc, const std::vector<std::uint64_t> &seeds, bool reshuffle, int nSims,
            double cPuct, const std::vector<const std::vector<double> *> &rootNoises, double noiseEps,
            const BatchLeafEval &batchEval, TerminalMode terminal, int vlBatch, double vlValue) {
        const std::size_t K = seeds.size();
        if (rootNoises.size() != K) throw std::invalid_argument("runTreesLockstep: one noise slot per tree");
        if (vlBatch < 1) throw std::invalid_argument("runTreesLockstep: virtual-loss batch must be >= 1");
        counters_.trees += static_cast<long long>(K);
        // One root bundle from the REAL bc, shared by every tree (runTree builds an identical one per tree;
        // the only difference is the features, present iff the root key is not yet in the nn cache).
        Bundle rootState = makeBundle(realBc, true);

        struct Tree {
            std::deque<Node> pool;
            double totalValue = 0.0;
        };
        std::vector<std::unique_ptr<Tree>> trees;
        trees.reserve(K);
        for (std::size_t k = 0; k < K; ++k) {
            trees.push_back(std::make_unique<Tree>());
            trees[k]->pool.emplace_back();
            trees[k]->pool[0].bc = cloneWithFreshRng(realBc, seeds[k], reshuffle);
            ++counters_.nodes;
        }

        auto afterSim = [&](std::size_t k, int simIdx) {
            if (rootNoises[k] != nullptr && simIdx == 0 && trees[k]->pool[0].expanded)
                applyRootNoise(trees[k]->pool[0], rootState, *rootNoises[k], noiseEps);
        };

        struct Pending { std::size_t tree; SimCursor cur; int row; };
        int done = 0;                                   // sims completed per tree (uniform across trees)
        int step = 0;
        vlInflight_ = 0;
        while (done < nSims) {
            // Step 0 is ONE descent per tree (the root expansion; noise follows). Later steps: up to B.
            const int per = step == 0 ? 1 : std::min(vlBatch, nSims - done);
            ++counters_.lockstepSteps;
            std::vector<Pending> pend;
            for (std::size_t k = 0; k < K; ++k) {                      // ascending tree index
                for (int d = 0; d < per; ++d) {                        // descents within a tree, in order
                    ++counters_.sims;
                    SimCursor cur;
                    double v = 0.0;
                    if (descend(trees[k]->pool, rootState, cPuct, nullptr, terminal, cur, v, step, vlValue)) {
                        trees[k]->totalValue += v;                     // finished w/o a net call: real backup, uncharged
                        afterSim(k, done + d);
                    } else {
                        if (cur.collision) ++counters_.vlCollisions;
                        else cur.node->pendStep = step;
                        vlCharge(cur.path, vlValue);
                        ++counters_.vlDescents;
                        pend.push_back(Pending{k, std::move(cur), -1});
                    }
                }
            }
            if (static_cast<long long>(pend.size()) > counters_.vlPendingMax)
                counters_.vlPendingMax = static_cast<long long>(pend.size());
            if (!pend.empty()) {
                // Batch composition: a pure function of the inputs. Rows = distinct nn keys at FIRST occurrence
                // in (tree, order); no padding.
                std::unordered_map<std::string, int> rowOf;
                std::vector<const Features *> feats;
                std::vector<const PyMask76 *> masks;
                for (auto &p : pend) {
                    const Bundle &b = *p.cur.state;
                    auto it = rowOf.find(b.key);
                    if (it == rowOf.end()) {
                        if (!b.feat) throw std::logic_error("PuctSearch: pending leaf without features");
                        p.row = static_cast<int>(feats.size());
                        rowOf.emplace(b.key, p.row);
                        feats.push_back(b.feat.get());
                        masks.push_back(&b.mask);
                    } else {
                        p.row = it->second;
                    }
                }
                const std::size_t nRows = feats.size();
                if (recordLog_) {
                    std::vector<LockstepLogEntry> log;
                    for (const auto &p : pend) {
                        std::string sig;
                        const Features &f = *feats[static_cast<std::size_t>(p.row)];
                        for (int sct = 0; sct < kNumFeatSections; ++sct) sig += sectionKey(f, static_cast<FeatSection>(sct));
                        log.push_back(LockstepLogEntry{static_cast<int>(p.tree), p.cur.state->key, p.row, std::move(sig)});
                    }
                    lockstepLog_.push_back(std::move(log));
                }
                ++counters_.batchCalls;
                counters_.leavesSubmitted += static_cast<long long>(pend.size());
                counters_.distinctRows += static_cast<long long>(nRows);
                if (static_cast<long long>(nRows) > counters_.maxBatch) counters_.maxBatch = static_cast<long long>(nRows);

                std::vector<LeafResult> results;
                if (mutation_ == 5) {                                       // TEST ONLY: reversed composition
                    std::vector<const Features *> rf(feats.rbegin(), feats.rend());
                    std::vector<const PyMask76 *> rm(masks.rbegin(), masks.rend());
                    results = batchEval(rf, rm);
                    if (results.size() == nRows) std::reverse(results.begin(), results.end());
                } else {
                    results = batchEval(feats, masks);
                }
                if (results.size() != nRows)
                    throw std::invalid_argument("batch leaf evaluator returned a wrong number of rows");

                std::vector<bool> inserted(nRows, false);
                for (auto &p : pend) {                                       // (tree, order)
                    Tree &t = *trees[p.tree];
                    Node &node = *p.cur.node;
                    const Bundle &b = *p.cur.state;
                    vlRelease(p.cur.path, vlValue);                          // this descent's charge comes off first
                    double v;
                    if (node.expanded) {
                        // a collision: the node was expanded by an earlier registered descent this step
                        v = results[static_cast<std::size_t>(p.row)].value;
                    } else if (!inserted[static_cast<std::size_t>(p.row)]) {
                        inserted[static_cast<std::size_t>(p.row)] = true;
                        ++counters_.nnMisses;
                        ++counters_.leafEvals;
                        const LeafResult &r = results[static_cast<std::size_t>(p.row)];
                        node.P = r.P;
                        node.mask = b.mask;
                        node.expanded = true;
                        nn_.emplace(b.key, NNEntry{b.mask, r.P, r.value});   // once; its own copy of P
                        v = r.value;
                    } else {
                        v = expand(node, b, nullptr);                        // shares the row: a cache hit
                    }
                    if (mutation_ == 3 && p.cur.path.empty())
                        p.cur.path.push_back({&node, puctSelectImpl(node.N.data(), node.W.data(), node.P.data(),
                                                                    node.mask.data(), cPuct, 0)});
                    backup(p.cur.path, v);
                    t.totalValue += v;
                    afterSim(p.tree, done);                                  // root noise only matters at sim 0
                }
            }
            if (vlInflight_ > counters_.vlInflightAfterStepMax) counters_.vlInflightAfterStepMax = vlInflight_;
            done += per;
            ++step;
        }

        std::vector<TreeResult> out(K);
        lastRootWs_.assign(K, std::array<float, PY_ACTION_SPACE>{});
        for (std::size_t k = 0; k < K; ++k) {
            out[k].rawN = trees[k]->pool[0].N;
            lastRootWs_[k] = trees[k]->pool[0].W;
            out[k].rootValue = nSims > 0 ? trees[k]->totalValue / static_cast<double>(nSims) : 0.0;
            if (static_cast<long long>(trees[k]->pool.size()) > counters_.peakTreeNodes)
                counters_.peakTreeNodes = static_cast<long long>(trees[k]->pool.size());
            if (debugChecks_)
                for (const auto &nd : trees[k]->pool)
                    if (nd.vlSum != 0) throw std::logic_error("PuctSearch: virtual loss left in flight after the search");
        }
        if (K > 0) lastRootW_ = lastRootWs_[K - 1];
        return out;
    }

}
