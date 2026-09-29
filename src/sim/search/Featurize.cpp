#include "sim/search/Featurize.h"
#include "sim/search/PyFeaturizerTables.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace sts;
using namespace sts::search;
namespace T = sts::search::pyfeat;

namespace {

    template <class A> constexpr int count(const A &a) { return static_cast<int>(std::size(a)); }

    [[noreturn]] void fail(const std::string &msg) { throw std::runtime_error("Featurize: " + msg); }

    // sts::InputState enumerator names in declaration order (the pybind enum's names; the Python
    // adapter maps the int to this name).
    constexpr const char *kInputStateEnum[] = {
        "EXECUTING_ACTIONS", "PLAYER_NORMAL", "CARD_SELECT", "CHOOSE_STANCE_ACTION",
        "CHOOSE_TOOLBOX_COLORLESS_CARD", "CHOOSE_EXHAUST_POTION_CARDS", "CHOOSE_GAMBLING_CARDS",
        "CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS", "CHOOSE_DISCARD_CARDS", "SCRY", "SELECT_ENEMY_ACTIONS",
        "FILL_RANDOM_POTIONS", "SHUFFLE_INTO_DRAW_BURN", "SHUFFLE_INTO_DRAW_VOID",
        "SHUFFLE_INTO_DRAW_DAZED", "SHUFFLE_INTO_DRAW_WOUND", "SHUFFLE_INTO_DRAW_SLIMED",
        "SHUFFLE_INTO_DRAW_ALL_STATUS", "SHUFFLE_CUR_CARD_INTO_DRAW", "SHUFFLE_DISCARD_TO_DRAW",
        "INITIAL_SHUFFLE", "CREATE_RANDOM_CARD_IN_HAND_POWER", "CREATE_RANDOM_CARD_IN_HAND_COLORLESS",
        "CREATE_RANDOM_CARD_IN_HAND_DEAD_BRANCH", "SELECT_CARD_IN_HAND_EXHAUST", "GENERATE_NILRY_CARDS",
        "EXHAUST_RANDOM_CARD_IN_HAND", "SELECT_STRANGE_SPOON_PROC", "SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON",
        "SELECT_WARPED_TONGS_CARD", "CREATE_ENCHIRIDION_POWER", "SELECT_CONFUSED_CARD_COST"};

    // Field orders this file hard-codes accessors for; verified against the generated header at init.
    constexpr const char *kMonsterNumericNames[] = {
        "artifact", "block_return", "choked", "corpse_explosion", "lock_on", "mark", "metallicize",
        "plated_armor", "poison", "regen", "shackled", "strength", "vulnerable", "weak"};
    constexpr const char *kMonsterBoolNames[] = {
        "asleep", "barricade", "minion", "minion_leader", "painful_stabs", "regrow", "shifting", "stasis"};

    template <class A> int findIn(const A &a, const std::string &s) {
        for (int i = 0; i < count(a); ++i) if (s == a[i]) return i;
        return -1;
    }

    template <class A, class B>
    void sameNames(const char *what, const A &a, const B &b) {
        if (count(a) != count(b))
            fail(std::string("stale PyFeaturizerTables.h: ") + what + " count " + std::to_string(count(a)) +
                 " != generated " + std::to_string(count(b)));
        for (int i = 0; i < count(a); ++i)
            if (std::strcmp(a[i], b[i]) != 0)
                fail(std::string("stale PyFeaturizerTables.h: ") + what + "[" + std::to_string(i) + "] '" + a[i] +
                     "' != generated '" + b[i] + "'");
    }

    struct Vocab {
        std::vector<short> cardIdx;       // per CardId ordinal: card_to_idx.get(cardStringIds[i], -1)
        std::vector<char> cardExcluded;   // per CardId ordinal: name in _CARD_VOCAB_EXCLUDED
        std::vector<short> mmid;          // per MonsterMoveId: mmid_to_idx.get(monsterMoveStrings[i], 0)
        std::vector<short> monId;         // per MonsterId:     monster_id_to_idx.get(monsterIdStrings[i], 0)
        std::vector<int> pow0, pow1;      // per MonsterId: unique-power channel (schema by id string) or -1
        std::vector<short> encounter;     // per MonsterEncounter: -1 = miss (Python raises)
        std::vector<short> task;          // per CardSelectTask:   -1 = miss (Python raises)
        std::vector<short> inputSlot;     // per InputState
        short stance[4];                  // per Stance: _stance_to_idx.get(stanceStrings[i], 0)
        short character[4];               // per CharacterClass: index in the 4-name list or -1 (silently none)
        int nCards, nEnc, nRelic, nPot, nTask, nInput, nStatus;
        // offsets into the scalar vector (era-1 encode_scalars_only layout)
        int oPlayer, oChar, oStatus, oStance, oOrbs, oRelicCnt, oRelic, oPotion, oMeta, oInput, oTurn, oSelect, total;

        Vocab() {
            // ---- stale-header guards: the sim's own name arrays must equal the generated ones ----
            sameNames("relicEnumNames", relicEnumNames, T::kRelicEnumNames);
            sameNames("potionEnumNames", potionEnumNames, T::kPotionEnumNames);
            sameNames("playerStatusEnumStrings", playerStatusEnumStrings, T::kPlayerStatusNames);
            sameNames("cardSelectTaskStrings", cardSelectTaskStrings, T::kCardSelectTaskNames);
            if (count(kInputStateEnum) != static_cast<int>(InputState::SELECT_CONFUSED_CARD_COST) + 1)
                fail("kInputStateEnum out of date with sts::InputState");
            if (count(kMonsterNumericNames) != count(T::kMonsterNumeric)) fail("monster numeric field count drifted");
            for (int i = 0; i < count(kMonsterNumericNames); ++i)
                if (std::strcmp(kMonsterNumericNames[i], T::kMonsterNumeric[i].name) != 0)
                    fail("monster numeric field order drifted at " + std::to_string(i));
            sameNames("monster bool fields", kMonsterBoolNames, T::kMonsterBool);
            const char *const pinned[] = {"FOCUS", "ARTIFACT", "DEXTERITY", "STRENGTH", "THE_BOMB"};
            const int pinnedAt[] = {55, 83, 84, 85, 86};
            for (int i = 0; i < 5; ++i)
                if (std::strcmp(T::kPlayerStatusNames[pinnedAt[i]], pinned[i]) != 0) fail("player status pin drifted");

            nCards = count(T::kCardVocab);
            constexpr int NC = count(cardStringIds);
            cardIdx.assign(NC, -1);
            cardExcluded.assign(NC, 0);
            std::unordered_map<std::string, int> cv;
            for (int i = 0; i < nCards; ++i) cv.emplace(T::kCardVocab[i], i);
            for (int i = 0; i < NC; ++i) {
                auto it = cv.find(cardStringIds[i]);
                if (it != cv.end()) cardIdx[i] = static_cast<short>(it->second);
                cardExcluded[i] = findIn(T::kCardExcluded, cardStringIds[i]) >= 0;
            }
            // Python: `.get(name, 0)` -- a DECLARED fallback index (unknown -> INVALID's slot 0).
            mmid.assign(count(monsterMoveStrings), 0);
            for (int i = 0; i < count(monsterMoveStrings); ++i) {
                int k = findIn(T::kMmidNames, monsterMoveStrings[i]);
                mmid[i] = static_cast<short>(k < 0 ? 0 : k);
            }
            constexpr int NM = count(monsterIdStrings);
            monId.assign(NM, 0);
            pow0.assign(NM, -1);
            pow1.assign(NM, -1);
            for (int i = 0; i < NM; ++i) {   // same `.get(name, 0)` fallback; index 0's string is "INVALID = 0"
                int k = findIn(T::kMonsterIdNames, monsterIdStrings[i]);
                monId[i] = static_cast<short>(k < 0 ? 0 : k);
                for (const auto &sc : T::kUniqueSchema)
                    if (std::strcmp(sc.monsterId, monsterIdStrings[i]) == 0) { pow0[i] = sc.pow0Chan; pow1[i] = sc.pow1Chan; }
            }
            nEnc = count(T::kEncounterNames);
            encounter.assign(count(monsterEncounterEnumNames), -1);
            for (int i = 0; i < count(monsterEncounterEnumNames); ++i)
                encounter[i] = static_cast<short>(findIn(T::kEncounterNames, monsterEncounterEnumNames[i]));
            nTask = count(T::kCardSelectTaskNames);
            task.assign(count(cardSelectTaskStrings), -1);
            for (int i = 0; i < count(cardSelectTaskStrings); ++i)
                task[i] = static_cast<short>(findIn(T::kCardSelectTaskNames, cardSelectTaskStrings[i]));
            nInput = count(T::kInputStateNames);
            inputSlot.assign(count(kInputStateEnum), static_cast<short>(nInput - 1));  // catch-all EXECUTING_ACTIONS slot
            for (int i = 0; i < count(kInputStateEnum); ++i) {
                int k = findIn(T::kInputStateNames, kInputStateEnum[i]);
                if (k >= 0) inputSlot[i] = static_cast<short>(k);
            }
            for (int i = 0; i < 4; ++i) {
                int k = findIn(T::kStanceNames, stanceStrings[i]);   // NOTE C++ stanceStrings has CALM/WRATH swapped (B21)
                stance[i] = static_cast<short>(k < 0 ? 0 : k);
                character[i] = static_cast<short>(findIn(T::kCharacterNames, characterClassEnumNames[i]));  // "SILENT" != "THE_SILENT": skipped, as in Python
            }
            nRelic = count(T::kRelicVocab);
            nPot = count(T::kPotionVocab);
            nStatus = count(T::kPlayerStatusNames);

            int o = 0;
            oPlayer = o; o += 9;
            oChar = o; o += 4;
            oStatus = o; o += nStatus;
            oStance = o; o += count(T::kStanceNames);
            oOrbs = o; o += 10 * 5;              // _MAX_ORB_SLOTS * len(_ORB_NAMES); never populated (get_state emits no orbs)
            oRelicCnt = o; o += 7;
            oRelic = o; o += nRelic;
            oPotion = o; o += 5 * nPot;
            oMeta = o; o += nEnc + 3;            // encounter one-hot, floor, stolen_gold, last_targeted (2 stasis dims omitted)
            oInput = o; o += nInput;
            oTurn = o; o += 4;
            oSelect = o; o += nTask + 1;
            total = o;
            if (total != T::kScalarDim) fail("scalar layout total " + std::to_string(total) + " != generated " + std::to_string(T::kScalarDim));
        }
    };

    const Vocab &vocab() {
        static const Vocab v;
        return v;
    }

    inline float f32(double x) { return static_cast<float>(x); }

    // combat_featurizer.card_index
    inline int cardIndex(const Vocab &v, const CardInstance &c, int numCardTypes) {
        int base = v.cardIdx[static_cast<int>(c.id)];
        if (base < 0 || base >= numCardTypes) return -1;
        return base + (c.isUpgraded() ? numCardTypes : 0);
    }

    template <class Pile>
    void pileIdx(const Vocab &v, const Pile &pile, int n, std::vector<std::int64_t> &out) {
        out.clear();
        for (const auto &c : pile) {
            int i = cardIndex(v, c, n);
            if (i >= 0) out.push_back(i);
        }
    }

    // CombatStateEncoder._stasis_card_fraction
    double stasisFraction(const Vocab &v, const CardInstance &sc) {
        if (sc.id == CardId::INVALID) return 0.0;                     // "NONE"
        const int o = static_cast<int>(sc.id);
        if (v.cardExcluded[o]) return 0.0;
        int idx = v.cardIdx[o];
        if (idx < 0) fail(std::string("unknown stasis card ") + cardStringIds[o]);
        return (idx + 1.0) / std::max(v.nCards, 1);
    }

    // The scalar vector: CombatStateEncoder.encode_scalars_only (era 1).
    void encodeScalars(const Vocab &v, const BattleContext &bc, std::vector<float> &s) {
        s.assign(v.total, 0.0f);
        const Player &p = bc.player;

        // player_scalars
        s[v.oPlayer + 0] = f32(static_cast<double>(p.curHp) / std::max(p.maxHp, 1));
        s[v.oPlayer + 1] = f32(p.maxHp / 999.0);
        s[v.oPlayer + 2] = f32(p.block / 999.0);
        s[v.oPlayer + 3] = f32(p.energy / 5.0);
        s[v.oPlayer + 4] = f32(static_cast<int>(p.energyPerTurn) / 5.0);
        s[v.oPlayer + 5] = f32(static_cast<int>(p.cardDrawPerTurn) / 10.0);
        s[v.oPlayer + 6] = f32(static_cast<int>(p.gold) / 999.0);
        s[v.oPlayer + 7] = f32(bc.ascension / 20.0);
        s[v.oPlayer + 8] = f32(bc.turn / 50.0);

        // character: silently skipped when not in Python's list (Silent -> "SILENT" != "THE_SILENT")
        {
            int k = v.character[static_cast<int>(p.cc)];
            if (k >= 0) s[v.oChar + k] = 1.0f;
        }

        // player_statuses
        {
            const int o = v.oStatus;
            s[o + 55] = f32(p.focus / T::kFocusNorm);
            s[o + 83] = f32(p.artifact / T::kArtifactNorm);
            s[o + 84] = f32(p.dexterity / T::kDexterityNorm);
            s[o + 85] = f32(p.strength / T::kStrengthNorm);
            s[o + 86] = f32(static_cast<int>(p.bomb3) / T::kTheBombNorm);   // statuses["THE_BOMB"] = bomb3
            for (const auto &w : T::kPlayerStatusMapped) {
                const auto st = static_cast<PlayerStatus>(w.idx);
                const double val = w.isBool ? (p.hasStatusRuntime(st) ? 1.0 : 0.0)
                                            : static_cast<double>(p.getStatusRuntime(st));
                s[o + w.idx] = f32(val / w.norm);
            }
        }

        // stance
        s[v.oStance + v.stance[static_cast<int>(p.stance)]] = 1.0f;

        // orbs: never populated (get_state has no "orbs"), matches Python's always-empty tuple.

        // relic_counters
        s[v.oRelicCnt + 0] = f32(static_cast<int>(p.happyFlowerCounter) / 3.0);
        s[v.oRelicCnt + 1] = f32(static_cast<int>(p.incenseBurnerCounter) / 6.0);
        s[v.oRelicCnt + 2] = f32(static_cast<int>(p.inkBottleCounter) / 10.0);
        s[v.oRelicCnt + 3] = f32(static_cast<int>(p.inserterCounter) / 2.0);
        s[v.oRelicCnt + 4] = f32(static_cast<int>(p.nunchakuCounter) / 10.0);
        s[v.oRelicCnt + 5] = f32(static_cast<int>(p.penNibCounter) / 10.0);
        s[v.oRelicCnt + 6] = f32(static_cast<int>(p.sundialCounter) / 3.0);

        // relic_presence: gc.relics == bc.initRelicBits; era 1 raises on a vocabulary miss
        for (int r = 0; r < count(T::kRelicEnumNames); ++r) {
            if (!((bc.initRelicBits[r >> 6] >> (r & 63)) & 1ULL)) continue;
            const int k = T::kRelicEnumToVocab[r];
            if (k < 0) fail(std::string("relic_presence: unknown relic ") + T::kRelicEnumNames[r]);
            s[v.oRelic + k] = 1.0f;
        }

        // potions
        for (int i = 0; i < bc.potionCapacity && i < 5; ++i) {
            const int k = T::kPotionEnumToVocab[static_cast<int>(bc.potions[i])];
            if (k < 0) fail(std::string("potions: unknown potion ") + potionEnumNames[static_cast<int>(bc.potions[i])]);
            s[v.oPotion + i * v.nPot + k] = 1.0f;
        }

        // combat_metadata (encounter one-hot, floor, stolen gold, last targeted; stasis dims omitted)
        {
            const int e = v.encounter[static_cast<int>(bc.encounter)];
            if (e < 0) fail(std::string("combat_metadata: unknown encounter ") + monsterEncounterEnumNames[static_cast<int>(bc.encounter)]);
            s[v.oMeta + e] = 1.0f;
            const int base = v.oMeta + v.nEnc;
            s[base + 0] = f32(bc.floorNum / 57.0);
            s[base + 1] = bc.requiresStolenGoldCheck() ? 1.0f : 0.0f;
            const int lt = static_cast<int>(p.lastTargetedMonster);
            s[base + 2] = f32(lt >= 0 ? lt / 5.0 : 0.0);
            // The stasis fractions are computed (and can raise) in Python although omitted from this vector.
            (void)stasisFraction(v, bc.cards.stasisCards[0]);
            (void)stasisFraction(v, bc.cards.stasisCards[1]);
        }

        // input_state
        s[v.oInput + v.inputSlot[static_cast<int>(bc.inputState)]] = 1.0f;

        // turn_tracking
        s[v.oTurn + 0] = f32(static_cast<int>(p.cardsPlayedThisTurn) / 20.0);
        s[v.oTurn + 1] = f32(static_cast<int>(p.attacksPlayedThisTurn) / 20.0);
        s[v.oTurn + 2] = f32(static_cast<int>(p.skillsPlayedThisTurn) / 20.0);
        s[v.oTurn + 3] = f32(static_cast<int>(p.cardsDiscardedThisTurn) / 20.0);

        // card_select: task only when the adapter reports CARD_SELECT, else "INVALID"; pick_count always
        {
            const auto &csi = bc.cardSelectInfo;
            int t = 0;   // "INVALID"
            if (bc.inputState == InputState::CARD_SELECT) {
                t = v.task[static_cast<int>(csi.cardSelectTask)];
                if (t < 0) fail("card_select: unknown task");
            }
            s[v.oSelect + t] = 1.0f;
            s[v.oSelect + v.nTask] = f32(csi.pickCount / 10.0);
        }
    }

    void putBytes(std::string &k, const void *p, std::size_t n) { k.append(static_cast<const char *>(p), n); }
    template <class V> void putVec(std::string &k, const V &v) {
        if (!v.empty()) putBytes(k, v.data(), v.size() * sizeof(typename V::value_type));
    }

}  // namespace

Features sts::search::featurize(const BattleContext &bc, const FeaturizerConfig &cfg) {
    if (cfg.era != 1) fail("only encoder_vocab_era 1 is emulated");
    if (cfg.monsterSlots != 5) fail("monsterSlots must be 5");
    const Vocab &v = vocab();
    Features f;
    encodeScalars(v, bc, f.scalars);

    // ---- hand (featurize_hand). NB: `retain` is never delivered by the adapter -> always 0.0 ----
    {
        const int S = cfg.handSlots;
        f.handIdx.assign(S, 0);
        f.handValid.assign(S, 0.0f);
        f.handScalars.assign(S * 3, 0.0f);
        const int nHand = bc.cards.cardsInHand;
        for (int i = 0; i < S && i < nHand; ++i) {
            const CardInstance &c = bc.cards.hand[i];
            const int ci = cardIndex(v, c, cfg.numCardTypes);
            f.handValid[i] = ci >= 0 ? 1.0f : 0.0f;
            f.handIdx[i] = std::max(ci, 0);
            f.handScalars[i * 3 + 0] = f32(static_cast<int>(c.costForTurn) / 3.0);
            f.handScalars[i * 3 + 1] = 0.0f;
            f.handScalars[i * 3 + 2] = f32(static_cast<int>(c.specialData) / 30.0);
        }
    }

    // ---- piles + Frozen Eye preview ----
    pileIdx(v, bc.cards.drawPile, cfg.numCardTypes, f.drawIdx);
    pileIdx(v, bc.cards.discardPile, cfg.numCardTypes, f.discardIdx);
    pileIdx(v, bc.cards.exhaustPile, cfg.numCardTypes, f.exhaustIdx);
    f.hasFrozenEye = ((bc.initRelicBits[T::kFrozenEyeRelicOrdinal >> 6] >> (T::kFrozenEyeRelicOrdinal & 63)) & 1ULL) != 0;
    if (f.hasFrozenEye) {
        f.previewIdx.assign(cfg.previewSlots, 0);
        f.previewValid.assign(cfg.previewSlots, 0.0f);
        int slot = 0;
        for (const auto &c : bc.cards.drawPile) {
            if (slot >= cfg.previewSlots) break;
            const int ci = cardIndex(v, c, cfg.numCardTypes);
            f.previewValid[slot] = ci >= 0 ? 1.0f : 0.0f;
            f.previewIdx[slot] = std::max(ci, 0);
            ++slot;
        }
    }

    // ---- card_select (DISCOVERY / CODEX pool options only; no num_card_types cap here) ----
    {
        f.selectIdx.assign(cfg.cardSelectSlots, 0);
        f.selectValid.assign(cfg.cardSelectSlots, 0.0f);
        if (bc.inputState == InputState::CARD_SELECT) {
            const auto &csi = bc.cardSelectInfo;
            const char *tn = cardSelectTaskStrings[static_cast<int>(csi.cardSelectTask)];
            if (std::strcmp(tn, "DISCOVERY") == 0 || std::strcmp(tn, "CODEX") == 0) {
                int slot = 0;
                for (int i = 0; i < 3; ++i) {
                    if (csi.cards[i] == CardId::INVALID) continue;   // adapter drops "INVALID" and compacts
                    if (slot >= cfg.cardSelectSlots) break;
                    const int idx = v.cardIdx[static_cast<int>(csi.cards[i])];
                    f.selectValid[slot] = idx >= 0 ? 1.0f : 0.0f;
                    f.selectIdx[slot] = std::max(idx, 0);
                    ++slot;
                }
            }
        }
    }

    // ---- stasis (featurize_stasis: no cap either) ----
    f.stasisIdx.assign(2, 0);
    f.stasisValid.assign(2, 0.0f);
    for (int i = 0; i < 2; ++i) {
        const CardInstance &sc = bc.cards.stasisCards[i];
        const int raw = sc.id == CardId::INVALID ? -1 : v.cardIdx[static_cast<int>(sc.id)];
        f.stasisValid[i] = raw >= 0 ? 1.0f : 0.0f;
        f.stasisIdx[i] = std::max(raw, 0);
    }

    // ---- monsters (featurize_monsters). Real slots first (<= 5), then empty-slot padding ----
    {
        constexpr int S = 5;
        f.monCurr.assign(S, 0); f.monH0.assign(S, 0); f.monH1.assign(S, 0); f.monMid.assign(S, 0);
        f.monTurnCol.assign(S, f32(std::min(bc.turn / 50.0, 1.0)));
        f.monStatuses.assign(S * 42, 0.0f);
        f.monScalars.assign(S * 8, 0.0f);
        const int cap1 = cfg.mmidCap - 1;
        const int nReal = std::min(bc.monsters.monsterCount, 5);
        constexpr int NUM = count(T::kMonsterNumeric), NCH = count(T::kUniqueChannels), NB = count(T::kMonsterBool);
        static_assert(NUM + NCH + NB == 42);
        for (int i = 0; i < S; ++i) {
            if (i >= nReal) continue;   // empty slot: "INVALID" ids -> index 0; every scalar 0 (make_empty_monster_slot)
            const Monster &m = bc.monsters.arr[i];
            const int mid = static_cast<int>(m.id);
            f.monCurr[i] = std::min<int>(v.mmid[static_cast<int>(m.moveHistory[0])], cap1);   // move_id == move_history_0
            f.monH0[i] = std::min<int>(v.mmid[static_cast<int>(m.moveHistory[0])], cap1);
            f.monH1[i] = std::min<int>(v.mmid[static_cast<int>(m.moveHistory[1])], cap1);
            f.monMid[i] = v.monId[mid];
            float *st = &f.monStatuses[i * 42];
            const double num[NUM] = {
                static_cast<double>(m.artifact), static_cast<double>(m.blockReturn), static_cast<double>(m.choked),
                static_cast<double>(m.corpseExplosion), static_cast<double>(m.lockOn), static_cast<double>(m.mark),
                static_cast<double>(m.metallicize), static_cast<double>(m.platedArmor), static_cast<double>(m.poison),
                static_cast<double>(m.regen), static_cast<double>(m.shackled), static_cast<double>(m.strength),
                static_cast<double>(m.vulnerable), static_cast<double>(m.weak)};
            for (int k = 0; k < NUM; ++k) st[k] = f32(num[k] / T::kMonsterNumeric[k].norm);
            for (int c = 0; c < NCH; ++c) {
                if (c == v.pow0[mid]) st[NUM + c] = f32(static_cast<double>(m.uniquePower0) / T::kUniqueChannels[c].norm);
                else if (c == v.pow1[mid]) st[NUM + c] = f32(static_cast<double>(m.uniquePower1) / T::kUniqueChannels[c].norm);
            }
            const bool bs[NB] = {m.hasStatus<MonsterStatus::ASLEEP>(), m.hasStatus<MonsterStatus::BARRICADE>(),
                                 m.hasStatus<MonsterStatus::MINION>(), m.hasStatus<MonsterStatus::MINION_LEADER>(),
                                 m.hasStatus<MonsterStatus::PAINFUL_STABS>(), m.hasStatus<MonsterStatus::REGROW>(),
                                 m.hasStatus<MonsterStatus::SHIFTING>(), m.hasStatus<MonsterStatus::STASIS>()};
            for (int k = 0; k < NB; ++k) st[NUM + NCH + k] = bs[k] ? 1.0f : 0.0f;

            // intent (get_state's monster dict)
            int intentDamage = 0, intentHits = 0;
            {
                DamageInfo di = m.getMoveBaseDamage(bc);
                if (di.attackCount > 0 && di.damage > 0) {
                    intentDamage = m.calculateDamageToPlayer(bc, di.damage);
                    intentHits = di.attackCount;
                }
            }
            float *sc = &f.monScalars[i * 8];
            sc[0] = f32(static_cast<double>(m.curHp) / std::max(m.maxHp, 1));
            sc[1] = f32(m.maxHp / 999.0);
            sc[2] = f32(m.block / 999.0);
            sc[3] = f32(intentDamage / 100.0);
            sc[4] = f32(intentHits / 10.0);
            sc[5] = m.isAlive() ? 1.0f : 0.0f;
            sc[6] = m.halfDead ? 1.0f : 0.0f;
            sc[7] = m.isEscaping() ? 1.0f : 0.0f;
        }
    }
    return f;
}

std::string sts::search::sectionKey(const Features &f, FeatSection s) {
    std::string k;
    switch (s) {
        case FeatSection::Scalars: putVec(k, f.scalars); break;
        case FeatSection::Hand: putVec(k, f.handIdx); putVec(k, f.handValid); putVec(k, f.handScalars); break;
        case FeatSection::Draw: {
            const char fe = f.hasFrozenEye ? 1 : 0;
            putBytes(k, &fe, 1);
            putVec(k, f.drawIdx);
            if (f.hasFrozenEye) { putVec(k, f.previewIdx); putVec(k, f.previewValid); }
            break;
        }
        case FeatSection::Discard: putVec(k, f.discardIdx); break;
        case FeatSection::Exhaust: putVec(k, f.exhaustIdx); break;
        case FeatSection::CardSelect: putVec(k, f.selectIdx); putVec(k, f.selectValid); break;
        case FeatSection::Stasis: putVec(k, f.stasisIdx); putVec(k, f.stasisValid); break;
        case FeatSection::Monster:
            putVec(k, f.monCurr); putVec(k, f.monH0); putVec(k, f.monH1); putVec(k, f.monMid);
            putVec(k, f.monTurnCol); putVec(k, f.monStatuses); putVec(k, f.monScalars);
            break;
    }
    return k;
}

std::vector<std::pair<std::string, std::vector<std::string>>> sts::search::vocabTables() {
    (void)vocab();   // also runs the stale-header guards
    auto v = [](const auto &a) { return std::vector<std::string>(std::begin(a), std::end(a)); };
    return {{"cards", v(T::kCardVocab)}, {"cards_excluded", v(T::kCardExcluded)}, {"relics", v(T::kRelicVocab)},
            {"potions", v(T::kPotionVocab)}, {"mmid", v(T::kMmidNames)}, {"monster_ids", v(T::kMonsterIdNames)},
            {"encounters", v(T::kEncounterNames)}, {"card_select_tasks", v(T::kCardSelectTaskNames)},
            {"input_states", v(T::kInputStateNames)}, {"stances", v(T::kStanceNames)},
            {"player_statuses", v(T::kPlayerStatusNames)}};
}
