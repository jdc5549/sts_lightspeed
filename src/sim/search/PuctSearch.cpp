// See include/sim/search/PuctSearch.h. Mirrors src/sts/agent/mcts.py line by line.
// No floating-point contraction anywhere in this file: the Python side never fuses.
#pragma GCC optimize("fp-contract=off")

#include "sim/search/PuctSearch.h"

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

    // _expand(node, state, net, nn_cache)
    double PuctSearch::expand(Node &node, const Bundle &state, const LeafEval &leafEval) {
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
        ++counters_.leafEvals;
        const LeafResult r = leafEval(*state.feat, state.mask);
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


    // _simulate(...). Nodes are pool indices; `pool[0]` is the root.
    double PuctSearch::simulate(std::deque<Node> &pool, const Bundle &rootState, double cPuct,
                                const LeafEval &leafEval, TerminalMode terminal) {
        struct PathEnt { Node *node; int action; };
        std::vector<PathEnt> path;
        auto doBackup = [&](double v) {
            for (auto it = path.rbegin(); it != path.rend(); ++it) {   // reversed(path)
                it->node->N[it->action] += 1;
                if (mutation_ == 4)
                    it->node->W[it->action] = static_cast<float>(static_cast<double>(it->node->W[it->action]) + v);
                else
                    it->node->W[it->action] = wAdd(it->node->W[it->action], v);
            }
        };
        Node *node = &pool[0];
        const Bundle *state = &rootState;

        while (true) {
            if (node->bc.outcome != Outcome::UNDECIDED) {          // terminal check
                const double v = terminalValue(node->bc, terminal);
                doBackup(v);
                return v;
            }
            if (!node->expanded) {                                  // leaf: expand
                if (!state->compat) { doBackup(0.0); return 0.0; }
                const double v = expand(*node, *state, leafEval);
                if (mutation_ == 3 && path.empty())
                    path.push_back({node, puctSelectImpl(node->N.data(), node->W.data(), node->P.data(),
                                                         node->mask.data(), cPuct, 0)});
                doBackup(v);
                return v;
            }
            const PyMask76 &mask = node->mask;                      // selection: nn_cache path
            bool any = false;
            for (bool m : mask) any = any || m;
            if (!any) { doBackup(0.0); return 0.0; }

            const int a = puctSelectImpl(node->N.data(), node->W.data(), node->P.data(), mask.data(), cPuct,
                                         mutation_);
            path.push_back({node, a});

            if (node->child[a] < 0) {
                const int parentIdx = static_cast<int>(node - &pool[0]);
                (void)parentIdx;
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
            // Noise after sim 0 expands the root (always expanded: the caller guarantees the root is
            // fixed-space compatible and not over, so sim 0 reaches _expand at the root).
            if (rootNoise != nullptr && simIdx == 0 && pool[0].expanded) {
                Node &root = pool[0];
                std::size_t i = 0;
                for (int a = 0; a < PY_ACTION_SPACE; ++a) {
                    if (!rootState.mask[a]) continue;
                    if (i >= rootNoise->size()) throw std::invalid_argument("root noise shorter than legal count");
                    root.P[a] = noiseMix(root.P[a], noiseEps, (*rootNoise)[i]);
                    ++i;
                }
                if (i != rootNoise->size()) throw std::invalid_argument("root noise length != legal count");
                if (mutation_ == 2) nn_[rootState.key].P = root.P;
            }
        }
        TreeResult res;
        res.rawN = pool[0].N;
        lastRootW_ = pool[0].W;
        res.rootValue = nSims > 0 ? totalValue / static_cast<double>(nSims) : 0.0;
        return res;
    }

}
