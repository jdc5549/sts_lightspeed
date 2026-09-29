#include "sim/search/StateHash.h"

namespace sts::search {

    std::uint64_t pyStateHash(const BattleContext &bc) {
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };

    const Player &p = bc.player;

    // --- Player core ---
    mix(static_cast<uint64_t>(p.curHp));
    mix(static_cast<uint64_t>(p.maxHp));
    mix(static_cast<uint64_t>(p.block));
    mix(static_cast<uint64_t>(p.energy));
    mix(static_cast<uint64_t>(p.energyPerTurn));
    mix(static_cast<uint64_t>(p.cardDrawPerTurn));
    mix(static_cast<uint64_t>(p.gold));
    mix(static_cast<uint64_t>(p.cc));
    mix(static_cast<uint64_t>(p.stance));
    mix(static_cast<uint64_t>(p.orbSlots));
    mix(static_cast<uint64_t>(p.lastTargetedMonster));
    mix(static_cast<uint64_t>(p.artifact));
    mix(static_cast<uint64_t>(p.dexterity));
    mix(static_cast<uint64_t>(p.focus));
    mix(static_cast<uint64_t>(p.strength));

    // --- Player status bits ---
    mix(p.statusBits0);
    mix(static_cast<uint64_t>(p.statusBits1));
    for (const auto &kv : p.statusMap) {
        mix(static_cast<uint64_t>(kv.first));
        mix(static_cast<uint64_t>(static_cast<uint16_t>(kv.second)));
    }

    // --- Relic counters ---
    mix(static_cast<uint64_t>(p.happyFlowerCounter));
    mix(static_cast<uint64_t>(p.incenseBurnerCounter));
    mix(static_cast<uint64_t>(p.inkBottleCounter));
    mix(static_cast<uint64_t>(p.inserterCounter));
    mix(static_cast<uint64_t>(p.nunchakuCounter));
    mix(static_cast<uint64_t>(p.penNibCounter));
    mix(static_cast<uint64_t>(p.sundialCounter));
    mix(p.haveUsedNecronomiconThisTurn ? 1ULL : 0ULL);

    // --- Internal counters ---
    mix(static_cast<uint64_t>(p.combustHpLoss));
    mix(static_cast<uint64_t>(p.devaFormEnergyPerTurn));
    mix(static_cast<uint64_t>(p.echoFormCardsDoubled));
    mix(static_cast<uint64_t>(p.panacheCounter));
    mix(static_cast<uint64_t>(p.bomb1));
    mix(static_cast<uint64_t>(p.bomb2));
    mix(static_cast<uint64_t>(p.bomb3));

    // --- Turn tracking ---
    mix(static_cast<uint64_t>(p.cardsPlayedThisTurn));
    mix(static_cast<uint64_t>(p.attacksPlayedThisTurn));
    mix(static_cast<uint64_t>(p.skillsPlayedThisTurn));
    mix(static_cast<uint64_t>(p.cardsDiscardedThisTurn));
    mix(p.orangePelletsCardTypesPlayed.to_ulong());

    // --- Combat metadata ---
    mix(static_cast<uint64_t>(bc.turn));
    mix(static_cast<uint64_t>(bc.ascension));
    mix(static_cast<uint64_t>(bc.floorNum));
    mix(static_cast<uint64_t>(bc.encounter));
    mix(static_cast<uint64_t>(bc.inputState));
    mix(bc.miscBits.to_ulong());

    // --- Potions ---
    mix(static_cast<uint64_t>(bc.potionCount));
    mix(static_cast<uint64_t>(bc.potionCapacity));
    for (int i = 0; i < bc.potionCapacity && i < 5; ++i) {
        mix(static_cast<uint64_t>(bc.potions[i]));
    }

    // --- Card select info ---
    {
        const auto &csi = bc.cardSelectInfo;
        mix(static_cast<uint64_t>(csi.cardSelectTask));
        mix(static_cast<uint64_t>(csi.pickCount));
        mix(csi.canPickZero ? 1ULL : 0ULL);
        mix(csi.canPickAnyNumber ? 1ULL : 0ULL);
        for (int i = 0; i < 3; ++i) {
            mix(static_cast<uint64_t>(csi.cards[i]));
        }
    }

    // --- Card piles ---
    auto hashCard = [&mix](const CardInstance &c) {
        mix(static_cast<uint64_t>(c.id));
        mix(c.upgraded ? 1ULL : 0ULL);
        mix(static_cast<uint64_t>(c.cost));
        mix(static_cast<uint64_t>(c.costForTurn));
        mix(static_cast<uint64_t>(static_cast<uint16_t>(c.specialData)));
    };
    mix(static_cast<uint64_t>(bc.cards.cardsInHand));
    for (int i = 0; i < bc.cards.cardsInHand; ++i) {
        hashCard(bc.cards.hand[i]);
    }
    mix(bc.cards.drawPile.size());
    for (const auto &c : bc.cards.drawPile)    { hashCard(c); }
    mix(bc.cards.discardPile.size());
    for (const auto &c : bc.cards.discardPile) { hashCard(c); }
    mix(bc.cards.exhaustPile.size());
    for (const auto &c : bc.cards.exhaustPile) { hashCard(c); }
    mix(static_cast<uint64_t>(bc.cards.stasisCards[0].id));
    mix(static_cast<uint64_t>(bc.cards.stasisCards[1].id));

    // --- Monsters ---
    mix(static_cast<uint64_t>(bc.monsters.monsterCount));
    for (int i = 0; i < bc.monsters.monsterCount && i < 5; ++i) {
        const Monster &m = bc.monsters.arr[i];
        mix(static_cast<uint64_t>(m.id));
        mix(static_cast<uint64_t>(m.curHp));
        mix(static_cast<uint64_t>(m.maxHp));
        mix(static_cast<uint64_t>(m.block));
        mix(static_cast<uint64_t>(m.moveHistory[0]));
        mix(static_cast<uint64_t>(m.moveHistory[1]));
        mix(static_cast<uint64_t>(m.strength));
        mix(static_cast<uint64_t>(m.vulnerable));
        mix(static_cast<uint64_t>(m.weak));
        mix(static_cast<uint64_t>(m.artifact));
        mix(static_cast<uint64_t>(m.poison));
        mix(static_cast<uint64_t>(m.metallicize));
        mix(static_cast<uint64_t>(m.platedArmor));
        mix(static_cast<uint64_t>(m.regen));
        mix(static_cast<uint64_t>(m.blockReturn));
        mix(static_cast<uint64_t>(m.choked));
        mix(static_cast<uint64_t>(m.corpseExplosion));
        mix(static_cast<uint64_t>(m.lockOn));
        mix(static_cast<uint64_t>(m.mark));
        mix(static_cast<uint64_t>(m.shackled));
        mix(static_cast<uint64_t>(m.uniquePower0));
        mix(static_cast<uint64_t>(m.uniquePower1));
        mix(m.statusBits);
        mix(m.halfDead ? 1ULL : 0ULL);
        mix(m.isEscapingB ? 1ULL : 0ULL);
        mix(m.escapeNext ? 1ULL : 0ULL);
        mix(static_cast<uint64_t>(m.miscInfo));
    }

    return h;
    }

}
