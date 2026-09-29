// C++ mirror of what the Python MCTS does between choosing an action index and the next node
// (PLAN-perf-cpp-search C.4d):
//   action = action_index_to_combat_action(idx, state)      (combat_action_space.py)
//   _execute_action(bc, action, state)  incl. _execute_card_select   (python_combat_runner.py)
//   _pump(bc)
// `state` in Python is the node's CACHED CombatState (frozen at first visit / from extract_cache),
// not necessarily an extraction of the live bc. So the conversion reads from a ConvCtx captured, at
// extraction time, from the bc the state was extracted from -- never re-derived from the live bc.
// Approximations/quirks of the Python path are reproduced, not fixed (decision 3).
#ifndef STS_LIGHTSPEED_APPLYACTION_H
#define STS_LIGHTSPEED_APPLYACTION_H

#include <vector>

#include "combat/BattleContext.h"

namespace sts::search {

    // Exactly what action_index_to_combat_action / _execute_card_select read from `state`.
    struct ConvCtx {
        // first i (< 5 monsters) with is_alive && !half_dead && !is_escaping, else 0
        int potionTarget = 0;
        // state.card_select_task: only non-None when the state's input state was CARD_SELECT.
        // -1 == None ('UNKNOWN' in Python); otherwise static_cast<int>(CardSelectTask)
        int cardSelectTask = -1;
        // SECRET_TECHNIQUE / SECRET_WEAPON only: indices into state.draw_pile whose CARD_PROPERTIES
        // type is SKILL / ATTACK respectively (empty for every other task)
        std::vector<int> secretIdx;

        bool operator==(const ConvCtx &o) const {
            return potionTarget == o.potionTarget && cardSelectTask == o.cardSelectTask && secretIdx == o.secretIdx;
        }
    };

    // Capture from the bc a CombatState is extracted from.
    ConvCtx makeConvCtx(const BattleContext &bc);

    // Convert + execute + pump. Throws std::invalid_argument for idx outside [0, 76) (Python ValueError);
    // C++ exceptions from play/end/potion/pump propagate (Python does not catch them there); those from
    // a card-select chooser or its resume_actions() are swallowed (Python logs and continues).
    // `swallowed` (optional) receives the number swallowed (0..2).
    void applyActionIndex(BattleContext &bc, int actionIdx, const ConvCtx &ctx, int *swallowed = nullptr);

    // Python _pump alone.
    void pump(BattleContext &bc);

}

#endif
