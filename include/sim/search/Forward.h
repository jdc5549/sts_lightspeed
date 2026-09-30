// Hand-rolled float32 forward of the micro net (CombatPolicyValueNetwork) from a C++ Features object
// (PLAN-cpp-inference B.1/B.2). Standalone: depends only on Featurize.h / PyLegality.h, never pybind, so the
// same translation unit is compiled into the forward bench (scripts/cpp/forward_bench, M.2).
//
// Numerics contract: every Linear is a plain sequential dot product PER OUTPUT (fixed ascending input order,
// accumulator starts at 0, bias added last); the loop is arranged input-major over contiguous output lanes so
// the compiler may vectorise ACROSS outputs but never reassociates within one (no fast-math, no FMA
// contraction: Forward.cpp is compiled with -ffp-contract=off -fno-fast-math, pinned in CMakeLists.txt).
// All accumulation is float32. LayerNorm: biased variance, 1/sqrt(var+eps). The only transcendental is
// std::exp(float) (libm expf) in the softmax.
#ifndef STS_LIGHTSPEED_FORWARD_H
#define STS_LIGHTSPEED_FORWARD_H

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sim/search/Featurize.h"
#include "sim/search/PyLegality.h"

namespace sts::search {

    struct ForwardOutput {
        std::array<float, PY_ACTION_SPACE> logits;   // masked logits: illegal slots = -1e8 (as net.forward_from_sections)
        std::array<float, PY_ACTION_SPACE> P;        // _post_process_row: softmax * mask, renorm / uniform-over-legal
        float value = 0.f;                           // value_head output[0]
    };

    class ForwardNet {
    public:
        explicit ForwardNet(const FeaturizerConfig &cfg) : cfg_(cfg) {}

        // Start (or restart) a weight load. `era` is the net's encoder_vocab_era; it must equal cfg.era or this
        // throws (an era-0 net's vocabularies/scalar layout differ from the featurizer's).
        void beginWeights(int era);
        // Throws std::runtime_error on an unknown name, a duplicate, a rank/shape mismatch that is decidable from
        // the FeaturizerConfig alone, or a size != prod(shape).
        void loadWeight(const std::string &name, const std::vector<std::int64_t> &shape, const float *data,
                        std::size_t count);
        // Throws listing ALL missing names, then checks every cross-tensor dimension. Builds the forward layout.
        void finalize();
        bool finalized() const { return finalized_; }
        // SHA-256 (hex) over the concatenated raw float bytes of every loaded tensor, in load order.
        std::string weightsDigest() const;

        // Requires finalize(). Scratch buffers are members: one thread per ForwardNet.
        ForwardOutput forward(const Features &f, const PyMask76 &mask);

        // The canonical (non-aliased) tensor names the net consumes, fixed part only (heads are pattern-named).
        static std::vector<std::string> fixedNames();

    private:
        struct Lin { int in = 0, out = 0; std::vector<float> wt; std::vector<float> b; };  // wt is [in][out]
        struct LN { int n = 0; std::vector<float> g, b; float eps = 1e-5f; };
        struct Tensor { std::vector<std::int64_t> shape; std::vector<float> data; };

        const Tensor &T(const std::string &name) const;
        Lin makeLin(const std::string &base) const;   // base = "x.0" -> x.0.weight, x.0.bias
        LN makeLN(const std::string &base) const;
        std::vector<Lin> makeHead(const std::string &prefix) const;
        static void linear(const Lin &l, const float *x, float *y);
        static void layerNorm(const LN &l, float *x);
        static void relu(float *x, int n);
        static const float *row(const std::vector<float> &t, int rows, int dim, std::int64_t idx, const char *what);

        FeaturizerConfig cfg_;
        bool begun_ = false, finalized_ = false;
        std::map<std::string, Tensor> tensors_;
        std::vector<std::string> loadOrder_;

        // layout (set by finalize)
        int d_ = 0, handTok_ = 0, handOut_ = 0, prevOut_ = 0, mmidD_ = 0, midD_ = 0, seqCtx_ = 0, monTok_ = 0,
            selOut_ = 0, stasOut_ = 0, scalarOut_ = 0, fuseOut_ = 0, scalarIn_ = 0, embRows_ = 0;
        Lin handCard_, handAgg_, prev_, seq0_, seq2_, mon_, sel_, stas_, sc0_, sc2lin_, fus_;
        LN handLN_, monLN_, sc1_, sc4_, fusLN_;
        float previewGate_ = 0.f;
        std::vector<float> emb_, mmidEmb_, midEmb_;   // [rows][dim] row-major
        int midRows_ = 0;
        std::vector<Lin> policy_, value_;

        // scratch
        std::vector<float> s_a_, s_b_, s_c_, s_d_, s_fuse_, s_hand_, s_nm_;
    };

}

#endif
