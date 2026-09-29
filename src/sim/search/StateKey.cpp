#include "sim/search/StateKey.h"
#include "sim/search/PyStateKeyFields.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

#include "constants/Cards.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include "constants/Potions.h"
#include "constants/PlayerStatusEffects.h"
#include "combat/CardSelectInfo.h"

using namespace sts;
using namespace sts::search;
namespace K = sts::search::pykey;
using PS = PlayerStatus;
using MS = MonsterStatus;

namespace {

    [[noreturn]] void fail(const std::string &msg) { throw std::runtime_error("StateKey: " + msg); }

    template <class A> constexpr int count(const A &a) { return static_cast<int>(std::size(a)); }

    // Writer. When `check` is set, every named field is verified against the generated list, in order:
    // top-level names against CombatState's list, names inside a card / monster against theirs.
    struct Seq { const char *const *names; int n; int next; };
    struct W {
        std::string s;
        bool check = false;
        Seq top{K::kCombatStateFields, count(K::kCombatStateFields), 0};
        Seq card{K::kCombatCardInstanceFields, count(K::kCombatCardInstanceFields), 0};
        Seq mon{K::kMonsterStateFields, count(K::kMonsterStateFields), 0};
        Seq *active = &top;

        void chk(const char *name) {
            if (!check) return;
            Seq &q = *active;
            if (q.next >= q.n || std::strcmp(q.names[q.next], name) != 0)
                fail(std::string("stale PyStateKeyFields.h: expected field '") + (q.next < q.n ? q.names[q.next] : "<end>") +
                     "' at position " + std::to_string(q.next) + ", serializer writes '" + name + "'");
            ++q.next;
        }
        void enter(Seq &q) { active = &q; q.next = 0; }
        void leave(const char *what) {
            if (check && active->next != active->n)
                fail(std::string("stale PyStateKeyFields.h: ") + what + " serializer wrote " +
                     std::to_string(active->next) + " fields, generated list has " + std::to_string(active->n));
            active = &top;
        }
        void varint(std::uint64_t v) {
            while (v >= 0x80) { s.push_back(static_cast<char>(v | 0x80)); v >>= 7; }
            s.push_back(static_cast<char>(v));
        }
        void i(const char *name, std::int64_t v) {
            chk(name);
            varint((static_cast<std::uint64_t>(v) << 1) ^ static_cast<std::uint64_t>(v >> 63));
        }
        void b(const char *name, bool v) { chk(name); s.push_back(v ? 1 : 0); }
        void len(const char *name, std::size_t v) { chk(name); varint(v); }
    };

    // Plain values of one card / monster slot, so an EMPTY slot and a real one share one writer.
    struct CardV { int id = 0; bool upgraded = false; int cost = 0, costForTurn = 0, specialData = 0; };
    struct MonV {
        int id = -1 /* empty slot: Python's "INVALID", distinct from every ordinal */, curHp = 0, maxHp = 0, block = 0, move = 0, h0 = 0, h1 = 0, intentDamage = 0, intentHits = 0;
        int strength = 0, vulnerable = 0, weak = 0, artifact = 0, poison = 0, metallicize = 0, platedArmor = 0,
            regen = 0, blockReturn = 0, choked = 0, corpseExplosion = 0, lockOn = 0, mark = 0, shackled = 0,
            uniquePower0 = 0, uniquePower1 = 0;
        bool asleep = false, barricade = false, minion = false, minionLeader = false, painfulStabs = false,
             regrow = false, shifting = false, stasis = false, alive = false, halfDead = false, escaping = false;
    };

    CardV cardV(const CardInstance &c) {
        CardV v;
        v.id = static_cast<int>(c.id); v.upgraded = c.isUpgraded(); v.cost = static_cast<int>(c.cost);
        v.costForTurn = static_cast<int>(c.costForTurn); v.specialData = static_cast<int>(c.specialData);
        return v;
    }

    MonV monV(const Monster &m, const BattleContext &bc) {
        MonV v;
        v.id = static_cast<int>(m.id); v.curHp = m.curHp; v.maxHp = m.maxHp; v.block = m.block;
        v.move = static_cast<int>(m.moveHistory[0]);   // move_id == move_history_0
        v.h0 = static_cast<int>(m.moveHistory[0]); v.h1 = static_cast<int>(m.moveHistory[1]);
        DamageInfo di = m.getMoveBaseDamage(bc);
        if (di.attackCount > 0 && di.damage > 0) {
            v.intentDamage = m.calculateDamageToPlayer(bc, di.damage);
            v.intentHits = di.attackCount;
        }
        v.strength = m.strength; v.vulnerable = m.vulnerable; v.weak = m.weak;
        v.artifact = static_cast<int>(m.artifact); v.poison = static_cast<int>(m.poison);
        v.metallicize = static_cast<int>(m.metallicize); v.platedArmor = static_cast<int>(m.platedArmor);
        v.regen = static_cast<int>(m.regen); v.blockReturn = static_cast<int>(m.blockReturn);
        v.choked = static_cast<int>(m.choked); v.corpseExplosion = static_cast<int>(m.corpseExplosion);
        v.lockOn = static_cast<int>(m.lockOn); v.mark = static_cast<int>(m.mark);
        v.shackled = static_cast<int>(m.shackled);
        v.uniquePower0 = m.uniquePower0; v.uniquePower1 = static_cast<int>(m.uniquePower1);
        v.asleep = m.hasStatus<MS::ASLEEP>(); v.barricade = m.hasStatus<MS::BARRICADE>();
        v.minion = m.hasStatus<MS::MINION>(); v.minionLeader = m.hasStatus<MS::MINION_LEADER>();
        v.painfulStabs = m.hasStatus<MS::PAINFUL_STABS>(); v.regrow = m.hasStatus<MS::REGROW>();
        v.shifting = m.hasStatus<MS::SHIFTING>(); v.stasis = m.hasStatus<MS::STASIS>();
        v.alive = m.isAlive(); v.halfDead = m.halfDead; v.escaping = m.isEscaping();
        return v;
    }

    void writeCard(W &w, const CardV &c) {
        w.enter(w.card);
        w.i("card_id", c.id); w.b("upgraded", c.upgraded); w.i("cost", c.cost);
        w.i("cost_for_turn", c.costForTurn); w.i("special_data", c.specialData);
        w.leave("CombatCardInstance");
    }

    void writeMonster(W &w, const MonV &m) {
        w.enter(w.mon);
        w.i("monster_id", m.id); w.i("current_hp", m.curHp); w.i("max_hp", m.maxHp); w.i("block", m.block);
        w.i("move_id", m.move); w.i("move_history_0", m.h0); w.i("move_history_1", m.h1);
        w.i("intent_damage", m.intentDamage); w.i("intent_hit_count", m.intentHits);
        w.i("strength", m.strength); w.i("vulnerable", m.vulnerable); w.i("weak", m.weak);
        w.i("artifact", m.artifact); w.i("poison", m.poison); w.i("metallicize", m.metallicize);
        w.i("plated_armor", m.platedArmor); w.i("regen", m.regen); w.i("block_return", m.blockReturn);
        w.i("choked", m.choked); w.i("corpse_explosion", m.corpseExplosion); w.i("lock_on", m.lockOn);
        w.i("mark", m.mark); w.i("shackled", m.shackled); w.i("unique_power0", m.uniquePower0);
        w.i("unique_power1", m.uniquePower1);
        w.b("asleep", m.asleep); w.b("barricade", m.barricade); w.b("minion", m.minion);
        w.b("minion_leader", m.minionLeader); w.b("painful_stabs", m.painfulStabs); w.b("regrow", m.regrow);
        w.b("shifting", m.shifting); w.b("stasis", m.stasis);
        w.b("is_alive", m.alive); w.b("half_dead", m.halfDead); w.b("is_escaping", m.escaping);
        w.leave("MonsterState");
    }

    // Name arrays must be pairwise distinct (else an ordinal would be finer than the Python string).
    template <class A> void distinct(const char *what, const A &a) {
        std::unordered_set<std::string> seen;
        for (int k = 0; k < count(a); ++k)
            if (!seen.insert(a[k]).second) fail(std::string(what) + " has a duplicate name: " + a[k]);
    }

    struct Init {
        int discovery = -1, codex = -1;
        Init() {
            distinct("cardStringIds", cardStringIds); distinct("monsterIdStrings", monsterIdStrings);
            distinct("monsterMoveStrings", monsterMoveStrings);
            distinct("monsterEncounterEnumNames", monsterEncounterEnumNames);
            distinct("cardSelectTaskStrings", cardSelectTaskStrings); distinct("stanceStrings", stanceStrings);
            distinct("characterClassEnumNames", characterClassEnumNames); distinct("potionEnumNames", potionEnumNames);
            if (std::strcmp(cardStringIds[0], "INVALID") != 0) fail("cardStringIds[0] != INVALID");
            for (int k = 0; k < count(cardStringIds); ++k)
                if (std::strcmp(cardStringIds[k], "NONE") == 0) fail("a card is named NONE (stasis sentinel)");
            for (int k = 0; k < count(cardSelectTaskStrings); ++k) {
                if (std::strcmp(cardSelectTaskStrings[k], "DISCOVERY") == 0) discovery = k;
                if (std::strcmp(cardSelectTaskStrings[k], "CODEX") == 0) codex = k;
            }
            if (discovery < 0 || codex < 0) fail("card-select task DISCOVERY/CODEX missing");
            // The empty monster slot's move ids are "INVALID" (ordinal 0).
            // (monsterIdStrings[0] is literally "INVALID = 0", so the empty slot's "INVALID" is written as ordinal -1.)
            for (int k = 0; k < count(monsterIdStrings); ++k)
                if (std::strcmp(monsterIdStrings[k], "INVALID") == 0) fail("a real monster id is named INVALID");
            if (std::strcmp(monsterMoveStrings[0], "INVALID") != 0) fail("monsterMoveStrings[0] != INVALID");
        }
    };
    const Init &init() { static const Init i; return i; }

    void writeAll(W &w, const BattleContext &bc) {
        const Init &I = init();
        const Player &p = bc.player;

        // --- Player core ---
        w.i("current_hp", p.curHp); w.i("max_hp", p.maxHp); w.i("block", p.block); w.i("energy", p.energy);
        w.i("energy_per_turn", static_cast<int>(p.energyPerTurn));
        w.i("card_draw_per_turn", static_cast<int>(p.cardDrawPerTurn));
        w.i("gold", static_cast<int>(p.gold));
        w.i("character", static_cast<int>(p.cc));            // characterClassEnumNames[cc], distinct
        w.i("ascension", bc.ascension); w.i("floor_num", bc.floorNum); w.i("turn", bc.turn);
        w.i("strength", p.strength); w.i("dexterity", p.dexterity); w.i("focus", p.focus); w.i("artifact", p.artifact);

        // --- Player statuses (expressions copied from get_state's `statuses` dict; bool/int per the adapter) ---
#include "StateKeyStatuses.inc"

        // --- Player internal counters / turn tracking / relic counters ---
        w.i("bomb1", static_cast<int>(p.bomb1)); w.i("bomb2", static_cast<int>(p.bomb2));
        w.i("bomb3", static_cast<int>(p.bomb3)); w.i("combust_hp_loss", static_cast<int>(p.combustHpLoss));
        w.i("deva_form_energy_per_turn", static_cast<int>(p.devaFormEnergyPerTurn));
        w.i("echo_form_cards_doubled", static_cast<int>(p.echoFormCardsDoubled));
        w.i("panache_counter", static_cast<int>(p.panacheCounter));
        w.b("have_used_necronomicon_this_turn", p.haveUsedNecronomiconThisTurn);
        w.i("cards_played_this_turn", static_cast<int>(p.cardsPlayedThisTurn));
        w.i("attacks_played_this_turn", static_cast<int>(p.attacksPlayedThisTurn));
        w.i("skills_played_this_turn", static_cast<int>(p.skillsPlayedThisTurn));
        w.i("cards_discarded_this_turn", static_cast<int>(p.cardsDiscardedThisTurn));
        w.b("orange_pellets_attack_played", (bool)p.orangePelletsCardTypesPlayed[0]);
        w.b("orange_pellets_skill_played", (bool)p.orangePelletsCardTypesPlayed[1]);
        w.b("orange_pellets_power_played", (bool)p.orangePelletsCardTypesPlayed[2]);
        w.i("happy_flower_counter", static_cast<int>(p.happyFlowerCounter));
        w.i("incense_burner_counter", static_cast<int>(p.incenseBurnerCounter));
        w.i("ink_bottle_counter", static_cast<int>(p.inkBottleCounter));
        w.i("inserter_counter", static_cast<int>(p.inserterCounter));
        w.i("nunchaku_counter", static_cast<int>(p.nunchakuCounter));
        w.i("pen_nib_counter", static_cast<int>(p.penNibCounter));
        w.i("sundial_counter", static_cast<int>(p.sundialCounter));

        // --- Orbs: get_state emits no "orbs" key, so Python's tuple is always empty (asserted by a unit test) ---
        w.i("orb_slots", static_cast<int>(p.orbSlots));
        w.len("orbs", 0);
        w.i("stance", static_cast<int>(p.stance));          // stanceStrings[stance], distinct

        // --- Piles ---
        w.len("hand", bc.cards.cardsInHand);
        for (int k = 0; k < bc.cards.cardsInHand; ++k) writeCard(w, cardV(bc.cards.hand[k]));
        w.len("draw_pile", bc.cards.drawPile.size());
        for (const auto &c : bc.cards.drawPile) writeCard(w, cardV(c));
        w.len("discard_pile", bc.cards.discardPile.size());
        for (const auto &c : bc.cards.discardPile) writeCard(w, cardV(c));
        w.len("exhaust_pile", bc.cards.exhaustPile.size());
        for (const auto &c : bc.cards.exhaustPile) writeCard(w, cardV(c));

        // --- Monsters: the adapter takes the first 5 and pads to exactly 5 with make_empty_monster_slot ---
        w.len("monsters", 5);
        const int nReal = std::min(bc.monsters.monsterCount, 5);
        for (int k = 0; k < 5; ++k) writeMonster(w, k < nReal ? monV(bc.monsters.arr[k], bc) : MonV{});
        w.i("monster_count", bc.monsters.monsterCount);

        // --- Potions: get_state lists min(capacity, 5); the name translation is asserted injective ---
        const int nPot = std::min<int>(bc.potionCapacity, 5);
        w.len("potions", nPot > 0 ? nPot : 0);
        for (int k = 0; k < nPot; ++k) w.varint(static_cast<std::uint64_t>(bc.potions[k]));   // Potion ordinal
        w.i("potion_capacity", bc.potionCapacity);

        // --- Relics: frozenset == gc.relics == initRelicBits (order-free by construction) ---
        w.chk("relics");
        for (int k = 0; k < 3; ++k)
            for (int b = 0; b < 8; ++b) w.s.push_back(static_cast<char>((bc.initRelicBits[k] >> (8 * b)) & 0xff));

        // --- Combat metadata ---
        w.i("encounter", static_cast<int>(bc.encounter));
        w.b("stolen_gold_check", bc.requiresStolenGoldCheck());
        w.i("last_targeted_monster", static_cast<int>(bc.player.lastTargetedMonster));
        w.i("stasis_card_0", static_cast<int>(bc.cards.stasisCards[0].id));   // INVALID <-> "NONE"
        w.i("stasis_card_1", static_cast<int>(bc.cards.stasisCards[1].id));
        // InputState: the adapter's int->name map falls back to PLAYER_NORMAL for an unregistered value.
        const int inRaw = static_cast<int>(bc.inputState);
        const int inState = (inRaw >= 0 && inRaw < K::kNumInputStates) ? inRaw : static_cast<int>(InputState::PLAYER_NORMAL);
        w.i("input_state", inState);

        // --- Card select (task / options only populated under CARD_SELECT, as the adapter does) ---
        const auto &csi = bc.cardSelectInfo;
        const bool inSelect = inState == static_cast<int>(InputState::CARD_SELECT);
        const int task = static_cast<int>(csi.cardSelectTask);
        w.chk("card_select_task");
        if (inSelect) { w.s.push_back(1); w.varint(static_cast<std::uint64_t>(task)); } else w.s.push_back(0);
        w.chk("card_select_options");
        int opts[3], nOpt = 0;
        if (inSelect && (task == I.discovery || task == I.codex))
            for (int k = 0; k < 3; ++k)
                if (csi.cards[k] != CardId::INVALID) opts[nOpt++] = static_cast<int>(csi.cards[k]);
        if (nOpt == 0) w.s.push_back(0);                      // None (an empty tuple is mapped to None)
        else { w.s.push_back(1); w.varint(nOpt); for (int k = 0; k < nOpt; ++k) w.varint(static_cast<std::uint64_t>(opts[k])); }
        w.i("card_select_pick_count", csi.pickCount);
        w.b("card_select_can_pick_zero", csi.canPickZero);
        w.b("card_select_can_pick_any", csi.canPickAnyNumber);
    }

    // Nested writers are exercised once on default values (a combat with no cards still verifies them).
    struct Checked {
        Checked() {
            init();
            W w; w.check = true;
            writeCard(w, CardV{});
            writeMonster(w, MonV{});
        }
    };

}

std::string sts::search::stateKey(const BattleContext &bc) {
    static const Checked checked;
    static bool topChecked = false;   // set only after a fully verified pass, so a throw re-checks next call
    W w;
    w.s.reserve(1024);
    if (!topChecked) {
        w.check = true;
        writeAll(w, bc);
        w.leave("CombatState");
        topChecked = true;
    } else {
        writeAll(w, bc);
    }
    return std::move(w.s);
}

std::vector<std::pair<std::string, std::vector<std::string>>> sts::search::stateKeyFieldNames() {
    auto v = [](const auto &a) { return std::vector<std::string>(std::begin(a), std::end(a)); };
    return {{"CombatState", v(K::kCombatStateFields)}, {"CombatCardInstance", v(K::kCombatCardInstanceFields)},
            {"MonsterState", v(K::kMonsterStateFields)}};
}
