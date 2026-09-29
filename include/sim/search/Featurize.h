// C++ EMULATION of the Python leaf-evaluation featurization (PLAN-perf-cpp-search C.3):
//   state = extract_combat_state(bc, gc)
//   scalars  = CombatStateEncoder.encode_scalars_only(state)          (665 float32, era 1)
//   sections = combat_featurizer.featurize_{hand,draw,pile,card_select,stasis,monsters}(state, ...)
// produced BITWISE identical (dtype, shape, padding, masks, live counts) from a BattleContext alone
// (gc.relics is read from BattleContext::initRelicBits). Python's lossiness and approximations are
// reproduced, not fixed (decision 3). Vocabularies are GENERATED (PyFeaturizerTables.h).
// Each section's exact key is its raw feature bytes (sectionKey).
#ifndef STS_LIGHTSPEED_FEATURIZE_H
#define STS_LIGHTSPEED_FEATURIZE_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "combat/BattleContext.h"

namespace sts::search {

    // The StructuredEncoderConfig values the Python featurizer takes (taken from the live encoder).
    struct FeaturizerConfig {
        int numCardTypes = 364;    // cfg.num_card_types: embedding half-size (NOT the flat vocab size)
        int handSlots = 10;
        int previewSlots = 5;      // cfg.frozen_eye_preview_slots
        int cardSelectSlots = 3;
        int monsterSlots = 5;      // must be 5 (the adapter always pads to 5)
        int mmidCap = 198;         // monster_encoder.mmid_embedding.num_embeddings
        int era = 1;               // encoder_vocab_era; only 1 is emulated
    };

    enum class FeatSection { Scalars = 0, Hand, Draw, Discard, Exhaust, CardSelect, Stasis, Monster };
    constexpr int kNumFeatSections = 8;

    // Field names/dtypes/shapes follow combat_featurizer's NamedTuples (int64 = torch.long).
    struct Features {
        std::vector<float> scalars;                          // (kScalarDim,)
        std::vector<std::int64_t> handIdx;                   // (handSlots,)
        std::vector<float> handValid;                        // (handSlots, 1)
        std::vector<float> handScalars;                      // (handSlots, 3)
        std::vector<std::int64_t> drawIdx;                   // (n,) live cards only
        bool hasFrozenEye = false;
        std::vector<std::int64_t> previewIdx;                // (previewSlots,) iff hasFrozenEye
        std::vector<float> previewValid;                     // (previewSlots, 1) iff hasFrozenEye
        std::vector<std::int64_t> discardIdx, exhaustIdx;    // (n,)
        std::vector<std::int64_t> selectIdx;                 // (cardSelectSlots,)
        std::vector<float> selectValid;                      // (cardSelectSlots, 1)
        std::vector<std::int64_t> stasisIdx;                 // (2,)
        std::vector<float> stasisValid;                      // (2, 1)
        std::vector<std::int64_t> monCurr, monH0, monH1, monMid;   // (5,)
        std::vector<float> monTurnCol;                       // (5, 1)
        std::vector<float> monStatuses;                      // (5, 42)
        std::vector<float> monScalars;                       // (5, 8)
    };

    // Throws std::runtime_error on a vocabulary miss where Python raises (relic / potion / encounter /
    // card-select task / unknown stasis card) and on an unsupported config.
    // The generated vocabularies compiled into this binary, by name (list of strings each), for the
    // Python-side import-time set check (src/sts/models/cpp_featurizer.py).
    std::vector<std::pair<std::string, std::vector<std::string>>> vocabTables();

    Features featurize(const BattleContext &bc, const FeaturizerConfig &cfg);

    // The section's raw feature bytes, prefixed by nothing (the scalar section's key equals
    // Python's `_scalar_key` bytes: encode_scalars_only(...).tobytes()).
    std::string sectionKey(const Features &f, FeatSection s);

}

#endif
