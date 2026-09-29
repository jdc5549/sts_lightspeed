#include "sim/search/GameLegality.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "constants/MonsterEncounters.h"
#include "constants/Potions.h"
#include "sim/search/ExpertKnowledge.h"

using namespace sts;
using namespace sts::search;

namespace {

    int g_mutation = 0;

    // Slots the apply side can act on for a card-select screen. Option index i lives at slot 61 + i:
    // hand tasks index the hand, pile tasks index the pile, SECRET_* index the i-th MATCHING draw card
    // (ApplyAction's ctx.secretIdx), DISCOVERY/CODEX index the three offered cards.
    void cardSelectSlots(const BattleContext &bc, PyMask76 &m) {
        const auto &csi = bc.cardSelectInfo;
        auto set = [&](int i) { if (i >= 0 && i < 10) m[61 + i] = true; };
        switch (csi.cardSelectTask) {
            case CardSelectTask::ARMAMENTS:
                for (int i = 0; i < bc.cards.cardsInHand; ++i) if (bc.cards.hand[i].canUpgrade()) set(i);
                break;
            case CardSelectTask::DUAL_WIELD:
                for (int i = 0; i < bc.cards.cardsInHand; ++i) {
                    const CardType t = bc.cards.hand[i].getType();
                    if (t == CardType::ATTACK || t == CardType::POWER) set(i);
                }
                break;
            case CardSelectTask::EXHAUST_ONE:
            case CardSelectTask::FORETHOUGHT:
            case CardSelectTask::WARCRY:
            case CardSelectTask::RECYCLE:
                for (int i = 0; i < bc.cards.cardsInHand; ++i) set(i);
                break;
            case CardSelectTask::HEADBUTT:
            case CardSelectTask::HOLOGRAM:
            case CardSelectTask::LIQUID_MEMORIES_POTION:
                for (int i = 0; i < static_cast<int>(bc.cards.discardPile.size()); ++i) set(i);
                break;
            case CardSelectTask::EXHUME:
                for (int i = 0; i < static_cast<int>(bc.cards.exhaustPile.size()); ++i)
                    if (bc.cards.exhaustPile[i].getId() != CardId::EXHUME) set(i);
                break;
            case CardSelectTask::SECRET_TECHNIQUE:
            case CardSelectTask::SECRET_WEAPON: {
                const CardType want = csi.cardSelectTask == CardSelectTask::SECRET_TECHNIQUE ? CardType::SKILL
                                                                                             : CardType::ATTACK;
                int k = 0;
                for (const auto &c : bc.cards.drawPile) if (c.getType() == want) set(k++);
                break;
            }
            case CardSelectTask::DISCOVERY:
            case CardSelectTask::CODEX:
                // CODEX also lets the player skip (BattleScumSearcher2's 4th action); the apply side cannot
                // apply it (chooseCodexCard has no skip and ApplyAction throws for opt >= 3), so it is not set.
                for (int i = 0; i < 3; ++i) if (csi.cards[i] != CardId::INVALID) set(i);
                break;
            default:
                // EXHAUST_MANY / GAMBLE are not fixed-space compatible (never searched); NIGHTMARE / MEDITATE /
                // SETUP / SEEK have no apply-side chooser: all-False, as in the emulation.
                break;
        }
    }

}

void search::setGameLegalityMutation(int m) { g_mutation = m; }

PyMask76 search::legalMask76Game(const BattleContext &bc) {
    PyMask76 m{};
    if (bc.inputState == InputState::CARD_SELECT) {
        cardSelectSlots(bc, m);
        return m;
    }
    if (bc.inputState != InputState::PLAYER_NORMAL) return m;

    // ---- cards: each hand slot on its own, through the game's canUse ----
    if (bc.isCardPlayAllowed()) {
        // TEST-ONLY mutation 1: number the slots by enumeration order (play-ordering sort + adjacent dedup)
        std::vector<int> slotOf(bc.cards.cardsInHand);
        for (int i = 0; i < bc.cards.cardsInHand; ++i) slotOf[i] = i;
        if (g_mutation == 1) {
            std::vector<std::pair<int, int>> order;
            for (int i = 0; i < bc.cards.cardsInHand; ++i) order.push_back({i, Expert::getPlayOrdering(bc.cards.hand[i].getId())});
            std::stable_sort(order.begin(), order.end(), [](auto a, auto b) { return a.second < b.second; });
            for (int s = 0; s < static_cast<int>(order.size()); ++s) slotOf[order[s].first] = s;
        }
        for (int h = 0; h < bc.cards.cardsInHand && h < 10; ++h) {
            const CardInstance &c = bc.cards.hand[h];
            const int slot = slotOf[h];
            if (c.requiresTarget()) {
                for (int t = 0; t < 5 && t < bc.monsters.monsterCount; ++t)
                    if (c.canUse(bc, t, false)) m[slot * 6 + t] = true;
            } else if (c.canUse(bc, 0, false)) {
                m[slot * 6 + 5] = true;
            }
        }
    }

    m[60] = true;   // EndTurn: always available on the player's normal input state

    // ---- potions: BattleScumSearcher2's usability rules for an occupied slot ----
    const bool hasValidTarget = bc.monsters.getTargetableCount() > 0;
    for (int i = 0; i < bc.potionCapacity && i < 5; ++i) {
        const Potion p = bc.potions[i];
        if (p == Potion::EMPTY_POTION_SLOT || p == Potion::INVALID) continue;
        if (p == Potion::FAIRY_POTION) continue;                                     // only ever discarded
        if (p == Potion::SMOKE_BOMB && !canSmokeBombEscape(bc.encounter)) continue;
        if (potionRequiresTarget(p) && !hasValidTarget) continue;
        m[71 + i] = true;
    }
    return m;
}
