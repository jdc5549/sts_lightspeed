#include "sim/search/PyLegality.h"
#include "sim/search/PyLegalityTables.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sts;
using namespace sts::search;

namespace {

    enum PyType : std::uint8_t { T_NONE = 0, T_ATTACK, T_SKILL, T_STATUS, T_CURSE };

    struct CardTables {
        std::vector<std::uint8_t> type;      // indexed by CardId
        std::vector<bool> requiresTarget;    // Python REQUIRES_TARGET membership (by string id)
        CardId blind = CardId::INVALID, trip = CardId::INVALID, clash = CardId::INVALID,
               slimed = CardId::INVALID, secretTech = CardId::INVALID, secretWeapon = CardId::INVALID;

        CardTables() {
            constexpr int N = static_cast<int>(sizeof(cardStringIds) / sizeof(cardStringIds[0]));
            std::unordered_map<std::string, int> idx;
            for (int i = 0; i < N; ++i) idx.emplace(cardStringIds[i], i);
            type.assign(N, T_NONE);
            requiresTarget.assign(N, false);
            auto find = [&](const char *s) { auto it = idx.find(s); return it == idx.end() ? -1 : it->second; };
            for (const char *s : pylegal::kRequiresTarget) { int i = find(s); if (i >= 0) requiresTarget[i] = true; }
            auto fill = [&](const char *const *b, const char *const *e, PyType t) {
                for (; b != e; ++b) { int i = find(*b); if (i >= 0) type[i] = t; }
            };
            fill(std::begin(pylegal::kTypeAttack), std::end(pylegal::kTypeAttack), T_ATTACK);
            fill(std::begin(pylegal::kTypeSkill), std::end(pylegal::kTypeSkill), T_SKILL);
            fill(std::begin(pylegal::kTypeStatus), std::end(pylegal::kTypeStatus), T_STATUS);
            fill(std::begin(pylegal::kTypeCurse), std::end(pylegal::kTypeCurse), T_CURSE);
            auto id = [&](const char *s) { int i = find(s); return i < 0 ? CardId::INVALID : static_cast<CardId>(i); };
            blind = id("Blind"); trip = id("Trip"); clash = id("Clash"); slimed = id("Slimed");
            secretTech = id("Secret Technique"); secretWeapon = id("Secret Weapon");
        }
    };

    const CardTables &tables() {
        static const CardTables t;
        return t;
    }

    inline PyType pyType(CardId id) { return static_cast<PyType>(tables().type[static_cast<int>(id)]); }

    // Python `_requires_target` / the mask's `requires_target`.
    inline bool pyRequiresTarget(const CardInstance &c) {
        const auto &t = tables();
        if ((c.id == t.blind || c.id == t.trip) && c.isUpgraded()) return false;
        return t.requiresTarget[static_cast<int>(c.id)];
    }

    bool taskIs(CardSelectTask t, const char *name) {
        return std::strcmp(cardSelectTaskStrings[static_cast<int>(t)], name) == 0;
    }

    template <std::size_t N>
    bool taskIn(CardSelectTask t, const char *const (&names)[N]) {
        for (const char *n : names) if (taskIs(t, n)) return true;
        return false;
    }

    constexpr const char *HAND_TASKS[] = {"ARMAMENTS", "EXHAUST_ONE", "EXHAUST_CARD", "DUAL_WIELD",
                                          "WARCRY", "RECYCLE", "FORETHOUGHT"};
    constexpr const char *DISCARD_TASKS[] = {"HEADBUTT", "DISCARD_TO_HAND", "HOLOGRAM", "LIQUID_MEMORIES_POTION"};
    constexpr const char *EXHAUST_TASKS[] = {"EXHUME"};
    constexpr const char *MULTI_TASKS[] = {"EXHAUST_MANY", "GAMBLE"};

    // First selected index of each CardSelectAction Python's _enumerate_card_select returns;
    // -1 stands for an action with EMPTY selected_indices (skipped by the mask).
    std::vector<int> cardSelectFirsts(const BattleContext &bc) {
        std::vector<int> out;
        const auto &csi = bc.cardSelectInfo;
        const auto task = csi.cardSelectTask;
        // Python: card_select_task is None only if the field were missing; here always set.
        // Pool options exist only for DISCOVERY/CODEX (adapter drops "INVALID" entries).
        int nOpts = 0;
        if (taskIs(task, "DISCOVERY") || taskIs(task, "CODEX")) {
            for (int i = 0; i < 3; ++i) if (csi.cards[i] != CardId::INVALID) ++nOpts;
        }
        const bool canPickZero = csi.canPickZero;

        if (taskIn(task, HAND_TASKS)) {
            const int n = bc.cards.cardsInHand;
            for (int i = 0; i < n; ++i) out.push_back(i);
            if (canPickZero) out.push_back(-1);
            if (out.empty()) out.push_back(0);
        } else if (taskIn(task, DISCARD_TASKS)) {
            const int n = static_cast<int>(bc.cards.discardPile.size());
            if (n == 0) out.push_back(-1);
            else for (int i = 0; i < n; ++i) out.push_back(i);
        } else if (taskIn(task, EXHAUST_TASKS)) {
            const int n = static_cast<int>(bc.cards.exhaustPile.size());
            if (n == 0) out.push_back(-1);
            else for (int i = 0; i < n; ++i) out.push_back(i);
        } else if (taskIn(task, MULTI_TASKS)) {
            const int n = bc.cards.cardsInHand;
            if (n == 0) { out.push_back(-1); return out; }
            constexpr std::size_t CAP = 32;
            if (canPickZero) out.push_back(-1);
            // itertools.combinations(range(n), k) order; only the first element matters, stop at cap.
            for (int k = 1; k <= n; ++k) {
                std::vector<int> c(k);
                for (int i = 0; i < k; ++i) c[i] = i;
                while (true) {
                    out.push_back(c[0]);
                    if (out.size() >= CAP) return out;
                    int i = k - 1;
                    while (i >= 0 && c[i] == i + n - k) --i;
                    if (i < 0) break;
                    ++c[i];
                    for (int j = i + 1; j < k; ++j) c[j] = c[j - 1] + 1;
                }
            }
        } else if (taskIs(task, "SECRET_TECHNIQUE") || taskIs(task, "SECRET_WEAPON")) {
            const PyType want = taskIs(task, "SECRET_TECHNIQUE") ? T_SKILL : T_ATTACK;
            int count = 0;
            for (const auto &c : bc.cards.drawPile) if (pyType(c.id) == want) ++count;
            if (count == 0) out.push_back(0);
            else for (int i = 0; i < std::min(count, 10); ++i) out.push_back(i);
        } else if (taskIs(task, "DISCOVERY") || taskIs(task, "CODEX")) {
            const int n = nOpts > 0 ? nOpts : 1;
            for (int i = 0; i < n; ++i) out.push_back(i);
        } else {
            // Unknown task: options are only ever populated for DISCOVERY/CODEX, so this is empty-tuple.
            out.push_back(-1);
        }
        return out;
    }

    bool isNormalOrSelect(const BattleContext &bc) {
        return bc.inputState == InputState::PLAYER_NORMAL || bc.inputState == InputState::CARD_SELECT;
    }

}

PyMask76 search::legalMask76(const BattleContext &bc) {
    PyMask76 m{};  // all False
    if (!isNormalOrSelect(bc)) return m;

    if (bc.inputState == InputState::CARD_SELECT) {
        for (int f : cardSelectFirsts(bc)) {
            if (f >= 0 && f < 10) m[61 + f] = true;
        }
        return m;
    }

    // ---- PLAYER_NORMAL ----
    const auto &t = tables();
    const Player &p = bc.player;

    // Alive monster slots over the first 5 (Python pads to 5 with dead slots).
    bool alive[5] = {false, false, false, false, false};
    int nMon = std::min(bc.monsters.monsterCount, 5);
    int firstAlive = 0;
    bool anyAlive = false;
    for (int i = 0; i < nMon; ++i) {
        const Monster &mo = bc.monsters.arr[i];
        alive[i] = mo.isAlive() && !mo.halfDead && !mo.isEscaping();
        if (alive[i] && !anyAlive) { anyAlive = true; firstAlive = i; }
    }
    (void) firstAlive;  // potion target is not part of the mask

    // Potions: slot legal unless sentinel, FairyPotion or SmokeBomb (Python _UNUSABLE_POTIONS).
    for (int i = 0; i < bc.potionCapacity && i < 5; ++i) {
        const Potion pot = bc.potions[i];
        if (pot == Potion::EMPTY_POTION_SLOT || pot == Potion::INVALID) continue;
        if (pot == Potion::FAIRY_POTION || pot == Potion::SMOKE_BOMB) continue;
        m[71 + i] = true;
    }

    const int energy = bc.player.energy;
    const bool entangled = p.hasStatus<PS::ENTANGLED>();
    const bool medicalKit = p.hasRelic<RelicId::MEDICAL_KIT>();
    const bool blueCandle = p.hasRelic<RelicId::BLUE_CANDLE>();

    const int handN = bc.cards.cardsInHand;
    bool hasClash = false, allAttacks = true;
    for (int i = 0; i < handN; ++i) {
        const CardId id = bc.cards.hand[i].id;
        if (id == t.clash) hasClash = true;
        if (pyType(id) != T_ATTACK) allAttacks = false;
    }
    const bool clashOk = hasClash ? allAttacks : false;

    bool drawHasSkill = false, drawHasAttack = false;
    for (const auto &c : bc.cards.drawPile) {
        const PyType ty = pyType(c.id);
        if (ty == T_SKILL) drawHasSkill = true;
        if (ty == T_ATTACK) drawHasAttack = true;
    }

    for (int i = 0; i < handN && i < 10; ++i) {
        const CardInstance &c = bc.cards.hand[i];
        const int cft = c.costForTurn;
        const PyType ty = pyType(c.id);
        // _can_play
        if (cft < -1) continue;
        if (ty == T_STATUS) { if (c.id != t.slimed && !medicalKit) continue; }
        else if (ty == T_CURSE) { if (!blueCandle) continue; }
        if (cft > energy) continue;
        if (entangled && ty == T_ATTACK) continue;

        if (c.id == t.clash && !clashOk) continue;
        if (c.id == t.secretTech && !drawHasSkill) continue;
        if (c.id == t.secretWeapon && !drawHasAttack) continue;

        if (pyRequiresTarget(c)) {
            for (int j = 0; j < 5; ++j) if (alive[j]) m[i * 6 + j] = true;
        } else {
            m[i * 6 + 5] = true;
        }
    }

    m[60] = true;  // EndTurn always appended
    return m;
}

bool search::fixedSpaceCompatible(const BattleContext &bc) {
    if (!isNormalOrSelect(bc)) return true;
    if (bc.inputState == InputState::CARD_SELECT) {
        if (taskIn(bc.cardSelectInfo.cardSelectTask, MULTI_TASKS)) return false;
        for (int f : cardSelectFirsts(bc)) if (f >= 10) return false;
    }
    return true;
}
