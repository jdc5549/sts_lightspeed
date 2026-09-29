#include "sim/search/ApplyAction.h"
#include "sim/search/PyLegality.h"

#include <exception>
#include <stdexcept>
#include <string>

using namespace sts;
using namespace sts::search;

namespace {

    constexpr int PLAY_CARD_END = 60;
    constexpr int END_TURN_IDX = 60;
    constexpr int CARD_SELECT_START = 61;
    constexpr int USE_POTION_START = 71;
    constexpr int ACTION_SPACE_SIZE = 76;
    constexpr int TARGETS_PER_SLOT = 6;
    constexpr int NO_TARGET_IDX = 5;
    constexpr int MAX_PUMP_ITERS = 200;

    // _PLAYER_INPUT_STATE_VALUES
    bool isPlayerInputState(InputState s) {
        switch (s) {
            case InputState::PLAYER_NORMAL:
            case InputState::CARD_SELECT:
            case InputState::CHOOSE_STANCE_ACTION:
            case InputState::CHOOSE_TOOLBOX_COLORLESS_CARD:
            case InputState::CHOOSE_EXHAUST_POTION_CARDS:
            case InputState::CHOOSE_GAMBLING_CARDS:
            case InputState::CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS:
            case InputState::CHOOSE_DISCARD_CARDS:
            case InputState::SCRY:
            case InputState::SELECT_CARD_IN_HAND_EXHAUST:
            case InputState::SELECT_WARPED_TONGS_CARD:
            case InputState::SELECT_CONFUSED_CARD_COST:
                return true;
            default:
                return false;
        }
    }

    // bc.resume_actions() binding
    void resumeActions(BattleContext &bc) {
        bc.inputState = InputState::EXECUTING_ACTIONS;
        bc.executeActions();
    }

    // bc.play_card binding
    void playCard(BattleContext &bc, int handIdx, int targetIdx) {
        if (handIdx < 0 || handIdx >= bc.cards.cardsInHand) return;
        const CardInstance &card = bc.cards.hand[handIdx];
        bc.addToBotCard(CardQueueItem(card, targetIdx, bc.player.energy));
        bc.inputState = InputState::EXECUTING_ACTIONS;
        bc.executeActions();
    }

    void endTurn(BattleContext &bc) {
        bc.endTurn();
        bc.inputState = InputState::EXECUTING_ACTIONS;
        bc.executeActions();
    }

    // bc.drink_potion binding
    void drinkPotion(BattleContext &bc, int slotIdx, int targetIdx) {
        if (slotIdx < 0 || slotIdx >= bc.potionCapacity) return;
        if (bc.potions[slotIdx] == Potion::EMPTY_POTION_SLOT) return;
        if (bc.potions[slotIdx] == Potion::FAIRY_POTION) return;
        if (bc.potions[slotIdx] == Potion::SMOKE_BOMB && !canSmokeBombEscape(bc.encounter)) return;
        bc.drinkPotion(slotIdx, targetIdx);
        bc.inputState = InputState::EXECUTING_ACTIONS;
        bc.executeActions();
    }

    // Bounds checks copied from the pybind lambdas (they `return` silently).
    inline bool handOk(const BattleContext &bc, int i) { return i >= 0 && i < bc.cards.cardsInHand; }

    // The body of the Python `try:` in _execute_card_select. May throw (the caller swallows).
    // Every branch mirrors the corresponding pybind lambda (bounds check included).
    void cardSelectChooser(BattleContext &bc, CardSelectTask task, int opt, const ConvCtx &ctx) {
        switch (task) {
            case CardSelectTask::ARMAMENTS:   if (handOk(bc, opt)) bc.chooseArmamentsCard(opt); break;
            case CardSelectTask::EXHAUST_ONE: if (handOk(bc, opt)) bc.chooseExhaustOneCard(opt); break;
            case CardSelectTask::DUAL_WIELD:  if (handOk(bc, opt)) bc.chooseDualWieldCard(opt); break;
            case CardSelectTask::WARCRY:      if (handOk(bc, opt)) bc.chooseWarcryCard(opt); break;
            case CardSelectTask::RECYCLE:     if (handOk(bc, opt)) bc.chooseRecycleCard(opt); break;
            case CardSelectTask::FORETHOUGHT: if (handOk(bc, opt)) bc.chooseForethoughtCard(opt); break;
            case CardSelectTask::HEADBUTT:    bc.chooseHeadbuttCard(opt); break;
            case CardSelectTask::HOLOGRAM:    bc.chooseDiscardToHandCard(opt, false); break;
            case CardSelectTask::LIQUID_MEMORIES_POTION: bc.chooseDiscardToHandCard(opt, true); break;
            case CardSelectTask::EXHUME:      bc.chooseExhumeCard(opt); break;
            case CardSelectTask::EXHAUST_MANY: {
                fixed_list<int, 10> fl; fl.push_back(opt); bc.chooseExhaustCards(fl); break;
            }
            case CardSelectTask::GAMBLE: {
                fixed_list<int, 10> fl; fl.push_back(opt); bc.chooseGambleCards(fl); break;
            }
            case CardSelectTask::DISCOVERY:
            case CardSelectTask::CODEX: {
                // Python: csi['cards'][idxs[0]] -> IndexError for opt >= 3 (swallowed by the caller)
                if (opt < 0 || opt >= 3) throw std::out_of_range("csi['cards'] index");
                const CardId id = bc.cardSelectInfo.cards[opt];
                if (task == CardSelectTask::DISCOVERY) bc.chooseDiscoveryCard(id);
                else bc.chooseCodexCard(id);
                break;
            }
            case CardSelectTask::SECRET_TECHNIQUE:
            case CardSelectTask::SECRET_WEAPON: {
                if (opt < static_cast<int>(ctx.secretIdx.size())) {
                    int drawIdx = ctx.secretIdx[opt];
                    bc.chooseDrawToHandCards(&drawIdx, 1);
                }
                break;
            }
            default:  // NIGHTMARE/MEDITATE/SETUP/SEEK (do nothing), INVALID / unknown (skip)
                break;
        }
    }

    void executeCardSelect(BattleContext &bc, int opt, const ConvCtx &ctx, int *swallowed) {
        // Python: task = state.card_select_task or 'UNKNOWN' -> unmatched -> skipped
        if (ctx.cardSelectTask >= 0) {
            try {
                cardSelectChooser(bc, static_cast<CardSelectTask>(ctx.cardSelectTask), opt, ctx);
            } catch (const std::exception &) {
                if (swallowed) ++*swallowed;
            }
        }
        try {
            resumeActions(bc);
        } catch (const std::exception &) {
            if (swallowed) ++*swallowed;
        }
    }

}

ConvCtx search::makeConvCtx(const BattleContext &bc) {
    ConvCtx c;
    for (int i = 0; i < bc.monsters.monsterCount && i < 5; ++i) {
        const Monster &m = bc.monsters.arr[i];
        if (m.isAlive() && !m.halfDead && !m.isEscaping()) { c.potionTarget = i; break; }
    }
    if (bc.inputState == InputState::CARD_SELECT) {
        const CardSelectTask t = bc.cardSelectInfo.cardSelectTask;
        c.cardSelectTask = static_cast<int>(t);
        if (t == CardSelectTask::SECRET_TECHNIQUE || t == CardSelectTask::SECRET_WEAPON) {
            const int want = t == CardSelectTask::SECRET_TECHNIQUE ? PY_TYPE_SKILL : PY_TYPE_ATTACK;
            int i = 0;
            for (const auto &card : bc.cards.drawPile) {
                if (pyCardType(card.id) == want) c.secretIdx.push_back(i);
                ++i;
            }
        }
    }
    return c;
}

void search::pump(BattleContext &bc) {
    for (int i = 0; i < MAX_PUMP_ITERS; ++i) {
        if (bc.outcome != Outcome::UNDECIDED) break;
        if (isPlayerInputState(bc.inputState)) break;
        if (bc.inputState == InputState::EXECUTING_ACTIONS) bc.executeActions();
        else resumeActions(bc);
    }
}

void search::applyActionIndex(BattleContext &bc, int idx, const ConvCtx &ctx, int *swallowed) {
    if (swallowed) *swallowed = 0;
    if (idx < 0 || idx >= ACTION_SPACE_SIZE)
        throw std::invalid_argument("Action index " + std::to_string(idx) + " out of range [0, 76)");

    if (idx < PLAY_CARD_END) {
        const int handIdx = idx / TARGETS_PER_SLOT;
        const int t = idx % TARGETS_PER_SLOT;
        playCard(bc, handIdx, t == NO_TARGET_IDX ? 0 : t);
    } else if (idx == END_TURN_IDX) {
        endTurn(bc);
    } else if (idx < USE_POTION_START) {
        executeCardSelect(bc, idx - CARD_SELECT_START, ctx, swallowed);
    } else {
        drinkPotion(bc, idx - USE_POTION_START, ctx.potionTarget);
    }
    pump(bc);
}
