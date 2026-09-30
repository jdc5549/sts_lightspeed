// See include/sim/search/Forward.h. Compiled with -ffp-contract=off -fno-fast-math (CMakeLists.txt).
#include "sim/search/Forward.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

namespace sts::search {

namespace {

// nn.LayerNorm's default eps. It is a module attribute, NOT a state_dict entry, so it cannot be loaded; the Python
// loader asserts every LayerNorm in the net has exactly this value.
constexpr float kLayerNormEps = 1e-5f;
constexpr float kMaskedLogit = -1e8f;

[[noreturn]] void fail(const std::string &msg) { throw std::runtime_error("ForwardNet: " + msg); }

std::string shapeStr(const std::vector<std::int64_t> &s) {
    std::string r = "(";
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + (s.size() == 1 ? ",)" : ")");
}

// ---- SHA-256 (FIPS 180-4) -------------------------------------------------------------------------------------
struct Sha256 {
    std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::uint8_t buf[64];
    std::size_t bufLen = 0;
    std::uint64_t total = 0;
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const std::uint8_t *p) {
        static const std::uint32_t K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[4*i]) << 24) | (std::uint32_t(p[4*i+1]) << 16) | (std::uint32_t(p[4*i+2]) << 8) | p[4*i+3];
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const std::uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void *data, std::size_t n) {
        const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
        total += n;
        while (n > 0) {
            const std::size_t take = std::min(n, std::size_t(64) - bufLen);
            std::memcpy(buf + bufLen, p, take);
            bufLen += take; p += take; n -= take;
            if (bufLen == 64) { block(buf); bufLen = 0; }
        }
    }
    std::string hex() {
        const std::uint64_t bits = total * 8;
        const std::uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (bufLen != 56) update(&zero, 1);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = std::uint8_t(bits >> (56 - 8 * i));
        update(len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

// Head tensors are pattern-named: "<policy|value>_head.<2k>.<weight|bias>". Returns true and fills k, isWeight.
bool parseHeadName(const std::string &name, std::string &prefix, int &k, bool &isWeight) {
    for (const char *pre : {"policy_head.", "value_head."}) {
        const std::string p(pre);
        if (name.compare(0, p.size(), p) != 0) continue;
        const std::string rest = name.substr(p.size());
        const auto dot = rest.find('.');
        if (dot == std::string::npos || dot == 0) return false;
        for (std::size_t i = 0; i < dot; ++i) if (rest[i] < '0' || rest[i] > '9') return false;
        const int idx = std::stoi(rest.substr(0, dot));
        const std::string tail = rest.substr(dot + 1);
        if (tail != "weight" && tail != "bias") return false;
        if (idx % 2 != 0) return false;   // odd indices are the ReLUs
        prefix = p.substr(0, p.size() - 1);
        k = idx / 2;
        isWeight = tail == "weight";
        return true;
    }
    return false;
}

struct FixedName { const char *name; int rank; };
const FixedName kFixed[] = {
    {"encoder.card_embedding.embedding.weight", 2},
    {"encoder.hand_encoder.card_mlp.0.weight", 2}, {"encoder.hand_encoder.card_mlp.0.bias", 1},
    {"encoder.hand_encoder.agg_mlp.0.weight", 2}, {"encoder.hand_encoder.agg_mlp.0.bias", 1},
    {"encoder.hand_encoder.agg_mlp.1.weight", 1}, {"encoder.hand_encoder.agg_mlp.1.bias", 1},
    {"encoder.draw_pile_encoder.preview_gate", 1},
    {"encoder.draw_pile_encoder.preview_mlp.0.weight", 2}, {"encoder.draw_pile_encoder.preview_mlp.0.bias", 1},
    {"encoder.monster_encoder.mmid_embedding.weight", 2}, {"encoder.monster_encoder.monster_id_embedding.weight", 2},
    {"encoder.monster_encoder.seq_mlp.0.weight", 2}, {"encoder.monster_encoder.seq_mlp.0.bias", 1},
    {"encoder.monster_encoder.seq_mlp.2.weight", 2}, {"encoder.monster_encoder.seq_mlp.2.bias", 1},
    {"encoder.monster_encoder.monster_mlp.0.weight", 2}, {"encoder.monster_encoder.monster_mlp.0.bias", 1},
    {"encoder.monster_encoder.monster_mlp.1.weight", 1}, {"encoder.monster_encoder.monster_mlp.1.bias", 1},
    {"encoder.card_select_encoder.mlp.0.weight", 2}, {"encoder.card_select_encoder.mlp.0.bias", 1},
    {"encoder.stasis_encoder.mlp.0.weight", 2}, {"encoder.stasis_encoder.mlp.0.bias", 1},
    {"encoder.scalar_encoder.mlp.0.weight", 2}, {"encoder.scalar_encoder.mlp.0.bias", 1},
    {"encoder.scalar_encoder.mlp.1.weight", 1}, {"encoder.scalar_encoder.mlp.1.bias", 1},
    {"encoder.scalar_encoder.mlp.3.weight", 2}, {"encoder.scalar_encoder.mlp.3.bias", 1},
    {"encoder.scalar_encoder.mlp.4.weight", 1}, {"encoder.scalar_encoder.mlp.4.bias", 1},
    {"encoder.fusion.0.weight", 2}, {"encoder.fusion.0.bias", 1},
    {"encoder.fusion.1.weight", 1}, {"encoder.fusion.1.bias", 1},
};

}  // namespace

std::vector<std::string> ForwardNet::fixedNames() {
    std::vector<std::string> v;
    for (const auto &f : kFixed) v.emplace_back(f.name);
    return v;
}

void ForwardNet::beginWeights(int era) {
    if (era != cfg_.era)
        fail("encoder-vocab era mismatch: the net's encoder_vocab_era is " + std::to_string(era) +
             " but this search's featurizer emulates era " + std::to_string(cfg_.era) +
             " (vocabularies and the scalar layout differ across eras)");
    if (cfg_.monsterSlots != 5) fail("monsterSlots must be 5, got " + std::to_string(cfg_.monsterSlots));
    tensors_.clear();
    loadOrder_.clear();
    finalized_ = false;
    begun_ = true;
}

void ForwardNet::loadWeight(const std::string &name, const std::vector<std::int64_t> &shape, const float *data,
                            std::size_t count) {
    if (!begun_) fail("loadWeight(" + name + ") before beginWeights(era)");
    int rank = -1;
    for (const auto &f : kFixed) if (name == f.name) { rank = f.rank; break; }
    std::string hp; int hk = 0; bool hw = false;
    if (rank < 0 && parseHeadName(name, hp, hk, hw)) rank = hw ? 2 : 1;
    if (rank < 0) fail("unknown weight name '" + name + "' (not a tensor the forward consumes; refusing to ignore it)");
    if (tensors_.count(name)) fail("weight '" + name + "' loaded twice");
    if (int(shape.size()) != rank)
        fail("weight '" + name + "' has shape " + shapeStr(shape) + ", expected rank " + std::to_string(rank));
    std::size_t n = 1;
    for (auto s : shape) { if (s <= 0) fail("weight '" + name + "' has non-positive dim in " + shapeStr(shape)); n *= std::size_t(s); }
    if (n != count) fail("weight '" + name + "': data has " + std::to_string(count) + " floats but shape " + shapeStr(shape) + " needs " + std::to_string(n));
    if (name == "encoder.card_embedding.embedding.weight" && shape[0] != 2 * std::int64_t(cfg_.numCardTypes))
        fail("weight '" + name + "' has " + std::to_string(shape[0]) + " rows, expected 2*num_card_types = " + std::to_string(2 * cfg_.numCardTypes));
    if (name == "encoder.monster_encoder.mmid_embedding.weight" && shape[0] != cfg_.mmidCap)
        fail("weight '" + name + "' has " + std::to_string(shape[0]) + " rows, expected mmid_cap = " + std::to_string(cfg_.mmidCap));
    if (name == "encoder.draw_pile_encoder.preview_gate" && shape[0] != 1)
        fail("weight '" + name + "' has shape " + shapeStr(shape) + ", expected (1,)");
    Tensor t;
    t.shape = shape;
    t.data.assign(data, data + count);
    tensors_.emplace(name, std::move(t));
    loadOrder_.push_back(name);
}

std::string ForwardNet::weightsDigest() const {
    Sha256 sh;
    for (const auto &name : loadOrder_) {
        const auto &d = tensors_.at(name).data;
        sh.update(d.data(), d.size() * sizeof(float));
    }
    return sh.hex();
}

const ForwardNet::Tensor &ForwardNet::T(const std::string &name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("internal: tensor '" + name + "' absent");
    return it->second;
}

ForwardNet::Lin ForwardNet::makeLin(const std::string &base) const {
    const Tensor &w = T(base + ".weight"), &b = T(base + ".bias");
    Lin l;
    l.out = int(w.shape[0]); l.in = int(w.shape[1]);
    if (int(b.shape[0]) != l.out)
        fail("'" + base + ".bias' has shape " + shapeStr(b.shape) + " but '" + base + ".weight' has shape " + shapeStr(w.shape));
    l.b = b.data;
    l.wt.resize(std::size_t(l.in) * l.out);   // transpose to [in][out]
    for (int o = 0; o < l.out; ++o)
        for (int i = 0; i < l.in; ++i) l.wt[std::size_t(i) * l.out + o] = w.data[std::size_t(o) * l.in + i];
    return l;
}

ForwardNet::LN ForwardNet::makeLN(const std::string &base) const {
    const Tensor &g = T(base + ".weight"), &b = T(base + ".bias");
    if (g.shape != b.shape) fail("LayerNorm '" + base + "' weight " + shapeStr(g.shape) + " vs bias " + shapeStr(b.shape));
    LN l;
    l.n = int(g.shape[0]); l.g = g.data; l.b = b.data; l.eps = kLayerNormEps;
    return l;
}

std::vector<ForwardNet::Lin> ForwardNet::makeHead(const std::string &prefix) const {
    std::vector<Lin> layers;
    for (int k = 0;; ++k) {
        const std::string base = prefix + "." + std::to_string(2 * k);
        if (!tensors_.count(base + ".weight")) break;
        layers.push_back(makeLin(base));
    }
    return layers;
}

void ForwardNet::finalize() {
    if (!begun_) fail("finalize() before beginWeights(era)");
    // 1. every missing fixed name, listed together.
    std::string missing;
    for (const auto &f : kFixed) if (!tensors_.count(f.name)) missing += std::string(missing.empty() ? "" : ", ") + f.name;
    for (const char *pre : {"policy_head", "value_head"}) {
        if (!tensors_.count(std::string(pre) + ".0.weight")) missing += std::string(missing.empty() ? "" : ", ") + pre + ".0.weight";
        if (!tensors_.count(std::string(pre) + ".0.bias")) missing += std::string(missing.empty() ? "" : ", ") + pre + ".0.bias";
    }
    // a head tensor whose weight/bias partner is absent (covers a dropped last layer that leaves the other half behind)
    for (const auto &name : loadOrder_) {
        std::string hp; int hk = 0; bool hw = false;
        if (!parseHeadName(name, hp, hk, hw)) continue;
        const std::string partner = hp + "." + std::to_string(2 * hk) + (hw ? ".bias" : ".weight");
        if (!tensors_.count(partner)) missing += std::string(missing.empty() ? "" : ", ") + partner;
    }
    if (!missing.empty()) fail("finalize: missing weights: " + missing);
    // 2. heads: layer indices must be contiguous 0,2,4,... with both weight and bias.
    for (const char *pre : {"policy_head", "value_head"}) {
        int nLayers = 0;
        while (tensors_.count(std::string(pre) + "." + std::to_string(2 * nLayers) + ".weight")) ++nLayers;
        for (const auto &name : loadOrder_) {
            std::string hp; int hk = 0; bool hw = false;
            if (parseHeadName(name, hp, hk, hw) && hp == pre && hk >= nLayers)
                fail("finalize: '" + name + "' is present but " + pre + " layers are not contiguous from index 0 (a lower "
                     "layer is missing)");
        }
        for (int k = 0; k < nLayers; ++k) {
            const std::string base = std::string(pre) + "." + std::to_string(2 * k);
            if (!tensors_.count(base + ".bias")) fail("finalize: missing weights: " + base + ".bias");
        }
    }

    // 3. build layers and check every cross-tensor dimension.
    const Tensor &E = T("encoder.card_embedding.embedding.weight");
    d_ = int(E.shape[1]); embRows_ = int(E.shape[0]); emb_ = E.data;
    const Tensor &MM = T("encoder.monster_encoder.mmid_embedding.weight");
    mmidD_ = int(MM.shape[1]); mmidEmb_ = MM.data;
    const Tensor &MI = T("encoder.monster_encoder.monster_id_embedding.weight");
    midD_ = int(MI.shape[1]); midRows_ = int(MI.shape[0]); midEmb_ = MI.data;
    previewGate_ = T("encoder.draw_pile_encoder.preview_gate").data[0];

    handCard_ = makeLin("encoder.hand_encoder.card_mlp.0");
    handAgg_ = makeLin("encoder.hand_encoder.agg_mlp.0");
    handLN_ = makeLN("encoder.hand_encoder.agg_mlp.1");
    prev_ = makeLin("encoder.draw_pile_encoder.preview_mlp.0");
    seq0_ = makeLin("encoder.monster_encoder.seq_mlp.0");
    seq2_ = makeLin("encoder.monster_encoder.seq_mlp.2");
    mon_ = makeLin("encoder.monster_encoder.monster_mlp.0");
    monLN_ = makeLN("encoder.monster_encoder.monster_mlp.1");
    sel_ = makeLin("encoder.card_select_encoder.mlp.0");
    stas_ = makeLin("encoder.stasis_encoder.mlp.0");
    sc0_ = makeLin("encoder.scalar_encoder.mlp.0");
    sc1_ = makeLN("encoder.scalar_encoder.mlp.1");
    sc2lin_ = makeLin("encoder.scalar_encoder.mlp.3");
    sc4_ = makeLN("encoder.scalar_encoder.mlp.4");
    fus_ = makeLin("encoder.fusion.0");
    fusLN_ = makeLN("encoder.fusion.1");
    policy_ = makeHead("policy_head");
    value_ = makeHead("value_head");

    handTok_ = handCard_.out; handOut_ = handAgg_.out; prevOut_ = prev_.out;
    seqCtx_ = seq2_.out; monTok_ = mon_.out; selOut_ = sel_.out; stasOut_ = stas_.out;
    scalarIn_ = sc0_.in; scalarOut_ = sc2lin_.out; fuseOut_ = fus_.out;

    auto need = [&](bool ok, const std::string &what) { if (!ok) fail("finalize: shape mismatch: " + what); };
    need(handCard_.in == d_ + 3, "hand card_mlp.0 in=" + std::to_string(handCard_.in) + " != card_embed_dim+3=" + std::to_string(d_ + 3));
    need(handAgg_.in == cfg_.handSlots * handTok_, "hand agg_mlp.0 in=" + std::to_string(handAgg_.in) + " != hand_slots*token=" + std::to_string(cfg_.handSlots * handTok_));
    need(handLN_.n == handOut_, "hand LayerNorm width " + std::to_string(handLN_.n) + " != " + std::to_string(handOut_));
    need(prev_.in == cfg_.previewSlots * d_, "preview_mlp.0 in=" + std::to_string(prev_.in) + " != preview_slots*card_embed_dim=" + std::to_string(cfg_.previewSlots * d_));
    need(seq0_.in == 3 * mmidD_ + midD_ + 1, "seq_mlp.0 in=" + std::to_string(seq0_.in) + " != 3*mmid+monster_id+1=" + std::to_string(3 * mmidD_ + midD_ + 1));
    need(seq2_.in == seq0_.out, "seq_mlp.2 in != seq_mlp.0 out");
    need(mon_.in == seqCtx_ + 3 + 2 + 42 + 3, "monster_mlp.0 in=" + std::to_string(mon_.in) + " != seq_context+50=" + std::to_string(seqCtx_ + 50));
    need(monLN_.n == monTok_, "monster LayerNorm width");
    need(sel_.in == cfg_.cardSelectSlots * d_, "card_select mlp.0 in=" + std::to_string(sel_.in) + " != slots*card_embed_dim=" + std::to_string(cfg_.cardSelectSlots * d_));
    need(stas_.in == 2 * d_, "stasis mlp.0 in=" + std::to_string(stas_.in) + " != 2*card_embed_dim");
    need(sc1_.n == sc0_.out, "scalar LayerNorm(1) width != scalar mlp.0 out");
    need(sc2lin_.in == sc0_.out, "scalar mlp.3 in != mlp.0 out");
    need(sc4_.n == scalarOut_, "scalar LayerNorm(4) width");
    const int fuseIn = scalarOut_ + handOut_ + (d_ + prevOut_) + d_ + d_ + cfg_.monsterSlots * monTok_ + selOut_ + stasOut_;
    need(fus_.in == fuseIn, "fusion.0 in=" + std::to_string(fus_.in) + " != sum of section widths=" + std::to_string(fuseIn));
    need(fusLN_.n == fuseOut_, "fusion LayerNorm width");
    for (int h = 0; h < 2; ++h) {
        const auto &L = h == 0 ? policy_ : value_;
        const char *nm = h == 0 ? "policy_head" : "value_head";
        need(!L.empty(), std::string(nm) + " has no layers");
        need(L[0].in == fuseOut_, std::string(nm) + ".0 in=" + std::to_string(L[0].in) + " != encoder output " + std::to_string(fuseOut_));
        for (std::size_t k = 1; k < L.size(); ++k) need(L[k].in == L[k - 1].out, std::string(nm) + " layer chain broken at " + std::to_string(2 * k));
    }
    need(policy_.back().out == PY_ACTION_SPACE, "policy_head output " + std::to_string(policy_.back().out) + " != action space " + std::to_string(PY_ACTION_SPACE));
    need(value_.back().out == 1, "value_head output " + std::to_string(value_.back().out) + " != 1");
    need(int(MI.shape.size()) == 2 && midRows_ > 0, "monster_id_embedding");

    s_fuse_.assign(fuseIn, 0.f);
    finalized_ = true;
}

// y[o] = (sum_i x[i] * W[o][i], ascending i, float32, starting from 0) + b[o]
void ForwardNet::linear(const Lin &l, const float *x, float *y) {
    const int out = l.out;
    for (int o = 0; o < out; ++o) y[o] = 0.f;
    for (int i = 0; i < l.in; ++i) {
        const float xi = x[i];
        const float *w = l.wt.data() + std::size_t(i) * out;
        for (int o = 0; o < out; ++o) y[o] += w[o] * xi;
    }
    for (int o = 0; o < out; ++o) y[o] += l.b[o];
}

void ForwardNet::layerNorm(const LN &l, float *x) {
    const int n = l.n;
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += x[i];
    const float mean = sum / float(n);
    float vs = 0.f;
    for (int i = 0; i < n; ++i) { const float c = x[i] - mean; vs += c * c; }
    const float var = vs / float(n);   // biased, as nn.LayerNorm
    const float rstd = 1.0f / std::sqrt(var + l.eps);
    for (int i = 0; i < n; ++i) x[i] = (x[i] - mean) * rstd * l.g[i] + l.b[i];
}

void ForwardNet::relu(float *x, int n) {
    for (int i = 0; i < n; ++i) x[i] = x[i] > 0.f ? x[i] : 0.f;   // max(x, 0): NaN-free inputs only
}

const float *ForwardNet::row(const std::vector<float> &t, int rows, int dim, std::int64_t idx, const char *what) {
    if (idx < 0 || idx >= rows)
        fail(std::string(what) + " index " + std::to_string(idx) + " out of range [0," + std::to_string(rows) + ")");
    return t.data() + std::size_t(idx) * dim;
}


// ---- feature-shape validation (shared by forward and forwardBatch) -------------------------------------------------
void ForwardNet::checkFeatures(const Features &f) const {
    if (!finalized_) fail("forward() before finalize()");
    const int H = cfg_.handSlots, P = cfg_.previewSlots, C = cfg_.cardSelectSlots, S = cfg_.monsterSlots;
    auto sz = [&](std::size_t got, std::size_t want, const char *what) {
        if (got != want) fail(std::string("Features.") + what + " has " + std::to_string(got) + " elements, expected " + std::to_string(want));
    };
    sz(f.scalars.size(), std::size_t(scalarIn_), "scalars");
    sz(f.handIdx.size(), std::size_t(H), "handIdx"); sz(f.handValid.size(), std::size_t(H), "handValid");
    sz(f.handScalars.size(), std::size_t(H) * 3, "handScalars");
    sz(f.selectIdx.size(), std::size_t(C), "selectIdx"); sz(f.selectValid.size(), std::size_t(C), "selectValid");
    sz(f.stasisIdx.size(), 2, "stasisIdx"); sz(f.stasisValid.size(), 2, "stasisValid");
    sz(f.monCurr.size(), std::size_t(S), "monCurr"); sz(f.monH0.size(), std::size_t(S), "monH0");
    sz(f.monH1.size(), std::size_t(S), "monH1"); sz(f.monMid.size(), std::size_t(S), "monMid");
    sz(f.monTurnCol.size(), std::size_t(S), "monTurnCol"); sz(f.monStatuses.size(), std::size_t(S) * 42, "monStatuses");
    sz(f.monScalars.size(), std::size_t(S) * 8, "monScalars");
    if (f.hasFrozenEye) { sz(f.previewIdx.size(), std::size_t(P), "previewIdx"); sz(f.previewValid.size(), std::size_t(P), "previewValid"); }
}

// ---- section layout (fusion order: scalars, hand, draw, discard, exhaust, monsters, card_select, stasis) ---------
int ForwardNet::sectionWidth(FeatSection s) const {
    switch (s) {
        case FeatSection::Scalars: return scalarOut_;
        case FeatSection::Hand: return handOut_;
        case FeatSection::Draw: return d_ + prevOut_;
        case FeatSection::Discard: case FeatSection::Exhaust: return d_;
        case FeatSection::Monster: return cfg_.monsterSlots * monTok_;
        case FeatSection::CardSelect: return selOut_;
        case FeatSection::Stasis: return stasOut_;
    }
    fail("sectionWidth: bad section");
}

int ForwardNet::sectionOffset(FeatSection s) const {
    int off = 0;
    for (FeatSection t : {FeatSection::Scalars, FeatSection::Hand, FeatSection::Draw, FeatSection::Discard,
                          FeatSection::Exhaust, FeatSection::Monster, FeatSection::CardSelect, FeatSection::Stasis}) {
        if (t == s) return off;
        off += sectionWidth(t);
    }
    fail("sectionOffset: bad section");
}

// ---- single-row section encoders ---------------------------------------------------------------------------------
// scalar: Linear, LN, ReLU, Linear, LN, ReLU
void ForwardNet::encScalars(const Features &f, float *out) {
    s_a_.resize(sc0_.out);
    linear(sc0_, f.scalars.data(), s_a_.data()); layerNorm(sc1_, s_a_.data()); relu(s_a_.data(), sc0_.out);
    linear(sc2lin_, s_a_.data(), out); layerNorm(sc4_, out); relu(out, scalarOut_);
}

// hand: per slot token = ReLU(Linear([emb*valid, scalars3])); concat; Linear, LN, ReLU
void ForwardNet::encHand(const Features &f, float *out) {
    const int H = cfg_.handSlots;
    s_a_.assign(std::size_t(H) * handTok_, 0.f);
    s_b_.resize(d_ + 3);
    for (int s = 0; s < H; ++s) {
        const float *e = row(emb_, embRows_, d_, f.handIdx[s], "hand card");
        const float v = f.handValid[s];
        for (int k = 0; k < d_; ++k) s_b_[k] = e[k] * v;
        for (int k = 0; k < 3; ++k) s_b_[d_ + k] = f.handScalars[std::size_t(s) * 3 + k];
        float *tok = s_a_.data() + std::size_t(s) * handTok_;
        linear(handCard_, s_b_.data(), tok); relu(tok, handTok_);
    }
    linear(handAgg_, s_a_.data(), out); layerNorm(handLN_, out); relu(out, handOut_);
}

// mean-pool of a pile's live cards (float32 sequential sum / N); empty pile -> zeros. Discard and exhaust are this
// same encoder; `what` only names the pile in an out-of-range error.
void ForwardNet::encPile(const std::vector<std::int64_t> &idx, float *out, const char *what) {
    for (int k = 0; k < d_; ++k) out[k] = 0.f;
    if (idx.empty()) return;
    for (std::int64_t id : idx) {
        const float *e = row(emb_, embRows_, d_, id, what);
        for (int k = 0; k < d_; ++k) out[k] += e[k];
    }
    const float n = float(idx.size());
    for (int k = 0; k < d_; ++k) out[k] /= n;
}

// draw: [mean(d), gate * ReLU(Linear(flatten(preview emb * valid)))] (zeros without Frozen Eye)
void ForwardNet::encDraw(const Features &f, float *out) {
    const int P = cfg_.previewSlots;
    encPile(f.drawIdx, out, "draw card");
    float *pv = out + d_;
    if (f.hasFrozenEye) {
        s_a_.resize(std::size_t(P) * d_);
        for (int s = 0; s < P; ++s) {
            const float *e = row(emb_, embRows_, d_, f.previewIdx[s], "preview card");
            const float v = f.previewValid[s];
            for (int k = 0; k < d_; ++k) s_a_[std::size_t(s) * d_ + k] = e[k] * v;
        }
        linear(prev_, s_a_.data(), pv); relu(pv, prevOut_);
        for (int k = 0; k < prevOut_; ++k) pv[k] = previewGate_ * pv[k];
    } else {
        for (int k = 0; k < prevOut_; ++k) pv[k] = 0.f;
    }
}

void ForwardNet::encMonsters(const Features &f, float *out) {
    const int S = cfg_.monsterSlots;
    s_a_.resize(seq0_.in); s_b_.resize(seq0_.out); s_c_.resize(seqCtx_); s_d_.resize(mon_.in);
    for (int s = 0; s < S; ++s) {
        monsterInput(f, s, s_a_.data());
        linear(seq0_, s_a_.data(), s_b_.data()); relu(s_b_.data(), seq0_.out);
        linear(seq2_, s_b_.data(), s_c_.data()); relu(s_c_.data(), seqCtx_);
        monsterTokenInput(f, s, s_c_.data(), s_d_.data());
        float *tok = out + std::size_t(s) * monTok_;
        linear(mon_, s_d_.data(), tok); layerNorm(monLN_, tok); relu(tok, monTok_);
    }
}

// seq_mlp input: [mmid(curr), mmid(h0), mmid(h1), monster_id emb, turn col]
void ForwardNet::monsterInput(const Features &f, int s, float *p) const {
    for (std::int64_t id : {f.monCurr[s], f.monH0[s], f.monH1[s]}) {
        const float *e = row(mmidEmb_, cfg_.mmidCap, mmidD_, id, "monster move id");
        for (int k = 0; k < mmidD_; ++k) *p++ = e[k];
    }
    const float *me = row(midEmb_, midRows_, midD_, f.monMid[s], "monster id");
    for (int k = 0; k < midD_; ++k) *p++ = me[k];
    *p = f.monTurnCol[s];
}

// monster_mlp input: [seq_context, scalars[:3], scalars[3:5], statuses(42), scalars[5:]]
void ForwardNet::monsterTokenInput(const Features &f, int s, const float *ctx, float *q) const {
    for (int k = 0; k < seqCtx_; ++k) *q++ = ctx[k];
    const float *sc = f.monScalars.data() + std::size_t(s) * 8;
    for (int k = 0; k < 5; ++k) *q++ = sc[k];
    const float *st = f.monStatuses.data() + std::size_t(s) * 42;
    for (int k = 0; k < 42; ++k) *q++ = st[k];
    for (int k = 5; k < 8; ++k) *q++ = sc[k];
}

void ForwardNet::encSelect(const Features &f, float *out) {
    const int C = cfg_.cardSelectSlots;
    s_a_.resize(std::size_t(C) * d_);
    for (int s = 0; s < C; ++s) {
        const float *e = row(emb_, embRows_, d_, f.selectIdx[s], "card-select card");
        const float v = f.selectValid[s];
        for (int k = 0; k < d_; ++k) s_a_[std::size_t(s) * d_ + k] = e[k] * v;
    }
    linear(sel_, s_a_.data(), out); relu(out, selOut_);
}

void ForwardNet::encStasis(const Features &f, float *out) {
    s_a_.resize(std::size_t(2) * d_);
    for (int s = 0; s < 2; ++s) {
        const float *e = row(emb_, embRows_, d_, f.stasisIdx[s], "stasis card");
        const float v = f.stasisValid[s];
        for (int k = 0; k < d_; ++k) s_a_[std::size_t(s) * d_ + k] = e[k] * v;
    }
    linear(stas_, s_a_.data(), out); relu(out, stasOut_);
}

void ForwardNet::encodeSection(FeatSection s, const Features &f, float *out) {
    switch (s) {
        case FeatSection::Scalars: encScalars(f, out); return;
        case FeatSection::Hand: encHand(f, out); return;
        case FeatSection::Draw: encDraw(f, out); return;
        case FeatSection::Discard: encPile(f.discardIdx, out, "discard card"); return;
        case FeatSection::Exhaust: encPile(f.exhaustIdx, out, "exhaust card"); return;
        case FeatSection::Monster: encMonsters(f, out); return;
        case FeatSection::CardSelect: encSelect(f, out); return;
        case FeatSection::Stasis: encStasis(f, out); return;
    }
    fail("encodeSection: bad section");
}

// ---- fusion, heads, post-processing ------------------------------------------------------------------------------
void ForwardNet::fusionStage(const float *fuse, float *nm) {
    linear(fus_, fuse, nm); layerNorm(fusLN_, nm); relu(nm, fuseOut_);
}

void ForwardNet::headsStage(const float *nm, std::vector<float> &lg, std::vector<float> &vl) {
    auto runHead = [&](const std::vector<Lin> &L, std::vector<float> &res) {
        std::vector<float> cur(nm, nm + fuseOut_), nxt;
        for (std::size_t k = 0; k < L.size(); ++k) {
            nxt.assign(L[k].out, 0.f);
            linear(L[k], cur.data(), nxt.data());
            if (k + 1 < L.size()) relu(nxt.data(), L[k].out);
            cur.swap(nxt);
        }
        res.swap(cur);
    };
    runHead(policy_, lg); runHead(value_, vl);
}

// _post_process_row: softmax(float logits) * mask, total > 1e-8 ? / total : uniform over legal
ForwardOutput ForwardNet::postProcess(const float *lg, float value, const PyMask76 &mask) {
    ForwardOutput out;
    for (int a = 0; a < PY_ACTION_SPACE; ++a) out.logits[a] = mask[a] ? lg[a] : kMaskedLogit;
    out.value = value;
    float mx = out.logits[0];
    for (int a = 1; a < PY_ACTION_SPACE; ++a) mx = std::max(mx, out.logits[a]);
    float ex[PY_ACTION_SPACE];
    float denom = 0.f;
    for (int a = 0; a < PY_ACTION_SPACE; ++a) { ex[a] = std::exp(out.logits[a] - mx); denom += ex[a]; }
    float total = 0.f;
    for (int a = 0; a < PY_ACTION_SPACE; ++a) { out.P[a] = mask[a] ? ex[a] / denom : 0.f; total += out.P[a]; }
    if (double(total) > 1e-8) {
        for (int a = 0; a < PY_ACTION_SPACE; ++a) out.P[a] = out.P[a] / total;
    } else {
        int nLegal = 0;
        for (int a = 0; a < PY_ACTION_SPACE; ++a) nLegal += mask[a] ? 1 : 0;
        for (int a = 0; a < PY_ACTION_SPACE; ++a) out.P[a] = nLegal > 0 && mask[a] ? 1.0f / float(nLegal) : 0.f;
    }
    return out;
}

ForwardOutput ForwardNet::fuseAndHeads(const float *fuse, const PyMask76 &mask) {
    s_nm_.resize(fuseOut_);
    fusionStage(fuse, s_nm_.data());
    std::vector<float> lg, vl;
    headsStage(s_nm_.data(), lg, vl);
    return postProcess(lg.data(), vl[0], mask);
}

ForwardOutput ForwardNet::forward(const Features &f, const PyMask76 &mask) {
    checkFeatures(f);
    float *fuse = s_fuse_.data();
    for (FeatSection s : {FeatSection::Scalars, FeatSection::Hand, FeatSection::Draw, FeatSection::Discard,
                          FeatSection::Exhaust, FeatSection::Monster, FeatSection::CardSelect, FeatSection::Stasis})
        encodeSection(s, f, fuse + sectionOffset(s));
    return fuseAndHeads(fuse, mask);
}

// ===== batched forward ===================================================================================================
// Every Linear is computed across the n rows with the weight row for input i loaded once and applied to all rows; each
// row's per-output accumulation is still `0 + w[0]*x[0] + w[1]*x[1] + ...` in ascending i, then + bias: the same
// operations in the same order as linear(), so every row is bit-identical to forward() on that row alone.
void ForwardNet::linearB(const Lin &l, const float *const *xs, float *const *ys, int n) {
    const int out = l.out;
    for (int b = 0; b < n; ++b) { float *y = ys[b]; for (int o = 0; o < out; ++o) y[o] = 0.f; }
    for (int i = 0; i < l.in; ++i) {
        const float *w = l.wt.data() + std::size_t(i) * out;
        for (int b = 0; b < n; ++b) {
            const float xi = xs[b][i];
            float *y = ys[b];
            for (int o = 0; o < out; ++o) y[o] += w[o] * xi;
        }
    }
    for (int b = 0; b < n; ++b) { float *y = ys[b]; for (int o = 0; o < out; ++o) y[o] += l.b[o]; }
}

std::vector<ForwardOutput> ForwardNet::forwardBatch(const std::vector<const Features *> &fs,
                                                    const std::vector<const PyMask76 *> &masks) {
    const int B = int(fs.size());
    if (masks.size() != fs.size()) fail("forwardBatch: " + std::to_string(fs.size()) + " feature rows but " + std::to_string(masks.size()) + " masks");
    std::vector<ForwardOutput> res;
    if (B == 0) return res;
    for (const Features *f : fs) checkFeatures(*f);
    const int H = cfg_.handSlots, S = cfg_.monsterSlots;
    const int FW = int(s_fuse_.size());
    std::vector<float> fuseAll(std::size_t(B) * FW, 0.f);
    auto fuseRow = [&](int b, FeatSection s) { return fuseAll.data() + std::size_t(b) * FW + sectionOffset(s); };

    std::vector<const float *> xs; std::vector<float *> ys;
    std::vector<float> A, Bv, Cv, Dv;

    // ---- scalars ----
    {
        A.assign(std::size_t(B) * sc0_.out, 0.f);
        xs.resize(B); ys.resize(B);
        for (int b = 0; b < B; ++b) { xs[b] = fs[b]->scalars.data(); ys[b] = A.data() + std::size_t(b) * sc0_.out; }
        linearB(sc0_, xs.data(), ys.data(), B);
        for (int b = 0; b < B; ++b) { layerNorm(sc1_, ys[b]); relu(ys[b], sc0_.out); xs[b] = ys[b]; ys[b] = fuseRow(b, FeatSection::Scalars); }
        linearB(sc2lin_, xs.data(), ys.data(), B);
        for (int b = 0; b < B; ++b) { layerNorm(sc4_, ys[b]); relu(ys[b], scalarOut_); }
    }
    // ---- hand: B*H token rows, then B aggregate rows ----
    {
        const int R = B * H;
        A.assign(std::size_t(R) * (d_ + 3), 0.f);
        Bv.assign(std::size_t(R) * handTok_, 0.f);
        xs.resize(R); ys.resize(R);
        for (int b = 0; b < B; ++b)
            for (int s = 0; s < H; ++s) {
                const Features &f = *fs[b];
                float *in = A.data() + std::size_t(b * H + s) * (d_ + 3);
                const float *e = row(emb_, embRows_, d_, f.handIdx[s], "hand card");
                const float v = f.handValid[s];
                for (int k = 0; k < d_; ++k) in[k] = e[k] * v;
                for (int k = 0; k < 3; ++k) in[d_ + k] = f.handScalars[std::size_t(s) * 3 + k];
                xs[b * H + s] = in; ys[b * H + s] = Bv.data() + std::size_t(b * H + s) * handTok_;
            }
        linearB(handCard_, xs.data(), ys.data(), R);
        for (int r = 0; r < R; ++r) relu(ys[r], handTok_);
        xs.resize(B); ys.resize(B);
        for (int b = 0; b < B; ++b) { xs[b] = Bv.data() + std::size_t(b) * H * handTok_; ys[b] = fuseRow(b, FeatSection::Hand); }
        linearB(handAgg_, xs.data(), ys.data(), B);
        for (int b = 0; b < B; ++b) { layerNorm(handLN_, ys[b]); relu(ys[b], handOut_); }
    }
    // ---- draw (mean + Frozen-Eye preview rows only for the rows that hold it), discard, exhaust ----
    {
        std::vector<int> eye;
        for (int b = 0; b < B; ++b) {
            float *o = fuseRow(b, FeatSection::Draw);
            encPile(fs[b]->drawIdx, o, "draw card");
            if (fs[b]->hasFrozenEye) eye.push_back(b);
            else for (int k = 0; k < prevOut_; ++k) o[d_ + k] = 0.f;
            encPile(fs[b]->discardIdx, fuseRow(b, FeatSection::Discard), "discard card");
            encPile(fs[b]->exhaustIdx, fuseRow(b, FeatSection::Exhaust), "exhaust card");
        }
        const int R = int(eye.size());
        if (R > 0) {
            const int P = cfg_.previewSlots;
            A.assign(std::size_t(R) * P * d_, 0.f);
            xs.resize(R); ys.resize(R);
            for (int r = 0; r < R; ++r) {
                const Features &f = *fs[eye[r]];
                float *in = A.data() + std::size_t(r) * P * d_;
                for (int s = 0; s < P; ++s) {
                    const float *e = row(emb_, embRows_, d_, f.previewIdx[s], "preview card");
                    const float v = f.previewValid[s];
                    for (int k = 0; k < d_; ++k) in[std::size_t(s) * d_ + k] = e[k] * v;
                }
                xs[r] = in; ys[r] = fuseRow(eye[r], FeatSection::Draw) + d_;
            }
            linearB(prev_, xs.data(), ys.data(), R);
            for (int r = 0; r < R; ++r) {
                relu(ys[r], prevOut_);
                for (int k = 0; k < prevOut_; ++k) ys[r][k] = previewGate_ * ys[r][k];
            }
        }
    }
    // ---- monsters: B*S rows ----
    {
        const int R = B * S;
        A.assign(std::size_t(R) * seq0_.in, 0.f);
        Bv.assign(std::size_t(R) * seq0_.out, 0.f);
        Cv.assign(std::size_t(R) * seqCtx_, 0.f);
        Dv.assign(std::size_t(R) * mon_.in, 0.f);
        xs.resize(R); ys.resize(R);
        for (int b = 0; b < B; ++b)
            for (int s = 0; s < S; ++s) {
                const int r = b * S + s;
                monsterInput(*fs[b], s, A.data() + std::size_t(r) * seq0_.in);
                xs[r] = A.data() + std::size_t(r) * seq0_.in; ys[r] = Bv.data() + std::size_t(r) * seq0_.out;
            }
        linearB(seq0_, xs.data(), ys.data(), R);
        for (int r = 0; r < R; ++r) { relu(ys[r], seq0_.out); xs[r] = ys[r]; ys[r] = Cv.data() + std::size_t(r) * seqCtx_; }
        linearB(seq2_, xs.data(), ys.data(), R);
        for (int r = 0; r < R; ++r) {
            relu(ys[r], seqCtx_);
            monsterTokenInput(*fs[r / S], r % S, ys[r], Dv.data() + std::size_t(r) * mon_.in);
            xs[r] = Dv.data() + std::size_t(r) * mon_.in;
            ys[r] = fuseRow(r / S, FeatSection::Monster) + std::size_t(r % S) * monTok_;
        }
        linearB(mon_, xs.data(), ys.data(), R);
        for (int r = 0; r < R; ++r) { layerNorm(monLN_, ys[r]); relu(ys[r], monTok_); }
    }
    // ---- card select, stasis ----
    {
        const int C = cfg_.cardSelectSlots;
        A.assign(std::size_t(B) * C * d_, 0.f);
        xs.resize(B); ys.resize(B);
        for (int b = 0; b < B; ++b) {
            const Features &f = *fs[b];
            float *in = A.data() + std::size_t(b) * C * d_;
            for (int s = 0; s < C; ++s) {
                const float *e = row(emb_, embRows_, d_, f.selectIdx[s], "card-select card");
                const float v = f.selectValid[s];
                for (int k = 0; k < d_; ++k) in[std::size_t(s) * d_ + k] = e[k] * v;
            }
            xs[b] = in; ys[b] = fuseRow(b, FeatSection::CardSelect);
        }
        linearB(sel_, xs.data(), ys.data(), B);
        for (int b = 0; b < B; ++b) relu(ys[b], selOut_);
        A.assign(std::size_t(B) * 2 * d_, 0.f);
        for (int b = 0; b < B; ++b) {
            const Features &f = *fs[b];
            float *in = A.data() + std::size_t(b) * 2 * d_;
            for (int s = 0; s < 2; ++s) {
                const float *e = row(emb_, embRows_, d_, f.stasisIdx[s], "stasis card");
                const float v = f.stasisValid[s];
                for (int k = 0; k < d_; ++k) in[std::size_t(s) * d_ + k] = e[k] * v;
            }
            xs[b] = in; ys[b] = fuseRow(b, FeatSection::Stasis);
        }
        linearB(stas_, xs.data(), ys.data(), B);
        for (int b = 0; b < B; ++b) relu(ys[b], stasOut_);
    }
    // ---- fusion ----
    std::vector<float> nm(std::size_t(B) * fuseOut_, 0.f);
    xs.resize(B); ys.resize(B);
    for (int b = 0; b < B; ++b) { xs[b] = fuseAll.data() + std::size_t(b) * FW; ys[b] = nm.data() + std::size_t(b) * fuseOut_; }
    linearB(fus_, xs.data(), ys.data(), B);
    for (int b = 0; b < B; ++b) { layerNorm(fusLN_, ys[b]); relu(ys[b], fuseOut_); }
    // ---- heads ----
    auto runHead = [&](const std::vector<Lin> &L, std::vector<std::vector<float>> &resv) {
        std::vector<std::vector<float>> cur(B), nxt(B);
        for (int b = 0; b < B; ++b) cur[b].assign(nm.data() + std::size_t(b) * fuseOut_, nm.data() + std::size_t(b + 1) * fuseOut_);
        for (std::size_t k = 0; k < L.size(); ++k) {
            for (int b = 0; b < B; ++b) { nxt[b].assign(L[k].out, 0.f); xs[b] = cur[b].data(); ys[b] = nxt[b].data(); }
            linearB(L[k], xs.data(), ys.data(), B);
            if (k + 1 < L.size()) for (int b = 0; b < B; ++b) relu(ys[b], L[k].out);
            cur.swap(nxt);
        }
        resv.swap(cur);
    };
    std::vector<std::vector<float>> lg, vl;
    runHead(policy_, lg); runHead(value_, vl);
    res.reserve(B);
    for (int b = 0; b < B; ++b) res.push_back(postProcess(lg[b].data(), vl[b][0], *masks[b]));
    return res;
}

}  // namespace sts::search
