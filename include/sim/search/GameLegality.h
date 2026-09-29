// The SIMULATOR's own legality over the same 76-slot layout as PyLegality.h (PLAN-perf-cpp-search S.3,
// `legality="game"`): 0-59 hand slot h * 6 + target t (t = 5: no target), 60 EndTurn, 61-70 card-select
// option, 71-75 potion slot. Where legalMask76 EMULATES the Python enumerator (approximations included),
// this asks the game: CardInstance::canUse per hand slot and per target, BattleContext::isCardPlayAllowed,
// Monster::isTargetable, the potion rules BattleScumSearcher2 uses, and the per-task selectable options.
// Slots are mapped by the real hand position / option index the apply side (ApplyAction.cpp) uses -- never by
// BattleScumSearcher2's enumeration order or its dedup of identical adjacent cards (an enumeration
// optimisation, not legality). fixedSpaceCompatible is unchanged, and non-PLAYER_NORMAL / CARD_SELECT input
// states are all-False, exactly as in the emulation.
#ifndef STS_LIGHTSPEED_GAMELEGALITY_H
#define STS_LIGHTSPEED_GAMELEGALITY_H

#include "sim/search/PyLegality.h"

namespace sts::search {

    PyMask76 legalMask76Game(const BattleContext &bc);

    // Test-only mutation for the S.3 index-vs-content guard: 0 = exact; 1 = hand slots are numbered by
    // BattleScumSearcher2's ENUMERATION order (play-ordering sort + adjacent-duplicate dedup) instead of the
    // real hand position. Never set outside tests.
    void setGameLegalityMutation(int m);

}

#endif
