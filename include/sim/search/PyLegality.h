// C++ emulation of the PYTHON legality path:
//   legal_action_mask(extract_combat_state(bc, gc))   (combat_action_space.py)
// over the 76-slot action space. Approximations in the Python enumerator are reproduced,
// not fixed (PLAN-perf-cpp-search decision 3). Layout: 0-59 card slot*6+target (5 = no target),
// 60 EndTurn, 61-70 card-select option, 71-75 potion slot.
#ifndef STS_LIGHTSPEED_PYLEGALITY_H
#define STS_LIGHTSPEED_PYLEGALITY_H

#include <array>

#include "combat/BattleContext.h"

namespace sts::search {

    constexpr int PY_ACTION_SPACE = 76;
    using PyMask76 = std::array<bool, PY_ACTION_SPACE>;

    // legal_action_mask(state); all-False unless input state is PLAYER_NORMAL / CARD_SELECT.
    PyMask76 legalMask76(const BattleContext &bc);

    // is_fixed_space_compatible(state).
    bool fixedSpaceCompatible(const BattleContext &bc);

    // CARD_PROPERTIES[card_id].get('type') as the Python enumerator sees it: 1 ATTACK, 2 SKILL, else 0/3/4.
    int pyCardType(CardId id);
    constexpr int PY_TYPE_ATTACK = 1;
    constexpr int PY_TYPE_SKILL = 2;

}

#endif
