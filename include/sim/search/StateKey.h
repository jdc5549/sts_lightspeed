// C++ KEY equivalent to Python's `CombatState` equality (PLAN-perf-cpp-search C.4c):
//   stateKey(bc1) == stateKey(bc2)  <=>  extract_combat_state(bc1, gc1) == extract_combat_state(bc2, gc2)
// for every state (gc-derived data comes from BattleContext::initRelicBits; nothing else is read).
// The Python search's `nn_cache` is keyed on the frozen CombatState, so the C++ tree must hit/miss
// on exactly this equivalence (the encoder-feature key is strictly coarser; see the C.4a census).
//
// Encoding: every CombatState / CombatCardInstance / MonsterState field is written, in dataclass
// order (field names are verified against the GENERATED PyStateKeyFields.h on first use), as a zig-zag
// varint (ints), one byte (bools) or a presence byte + payload (Optional). Every variable-length
// collection is length-prefixed. There are no strings and no floats: name-valued fields (card / monster /
// move / potion / encounter / task / stance / character) are written as their enum ordinal, which is
// injective because the name arrays are verified pairwise-distinct at first use (and the Python
// relic/potion translations are asserted injective by scripts/gen_py_state_key_fields.py).
// `relics` (a frozenset in Python) is the three initRelicBits words -- canonically ordered by construction.
#ifndef STS_LIGHTSPEED_STATEKEY_H
#define STS_LIGHTSPEED_STATEKEY_H

#include <string>
#include <vector>

#include "combat/BattleContext.h"

namespace sts::search {

    std::string stateKey(const BattleContext &bc);   // throws std::runtime_error on a stale field list

    // The field-name sequences the serializer writes (== the generated header's), for the Python
    // staleness test: {"CombatState": [...], "CombatCardInstance": [...], "MonsterState": [...]}.
    std::vector<std::pair<std::string, std::vector<std::string>>> stateKeyFieldNames();

}

#endif
