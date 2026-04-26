//
// Created by keega on 9/16/2021.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/search/SimpleAgent.h"
#include "sim/RandomAgent.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/Neow.h"

#include "slaythespire.h"
#include "combat/BattleContext.h"


using namespace sts;

PYBIND11_MODULE(slaythespire, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("play", &sts::py::play, "play Slay the Spire Console");
    m.def("get_seed_str", &SeedHelper::getString, "gets the integral representation of seed string used in the game ui");
    m.def("get_seed_long", &SeedHelper::getLong, "gets the seed string representation of an integral seed");
    m.def("getNNInterface", &sts::NNInterface::getInstance, "gets the NNInterface object");

    pybind11::class_<NNInterface> nnInterface(m, "NNInterface");
    nnInterface.def("getObservation", &NNInterface::getObservation, "get observation array given a GameContext")
        .def("getObservationMaximums", &NNInterface::getObservationMaximums, "get the defined maximum values of the observation space")
        .def_property_readonly("observation_space_size", []() { return NNInterface::observation_space_size; });

    pybind11::class_<search::ScumSearchAgent2> agent(m, "Agent");
    agent.def(pybind11::init<>());
    agent.def_readwrite("simulation_count_base", &search::ScumSearchAgent2::simulationCountBase, "number of simulations the agent uses for monte carlo tree search each turn")
        .def_readwrite("boss_simulation_multiplier", &search::ScumSearchAgent2::bossSimulationMultiplier, "bonus multiplier to the simulation count for boss fights")
        .def_readwrite("pause_on_card_reward", &search::ScumSearchAgent2::pauseOnCardReward, "causes the agent to pause so as to cede control to the user when it encounters a card reward choice")
        .def_readwrite("print_logs", &search::ScumSearchAgent2::printLogs, "when set to true, the agent prints state information as it makes actions")
        .def("playout", &search::ScumSearchAgent2::playout);

    // Standalone helpers for running battles or full games with alternative agents.
    //
    // play_battle_simple / play_battle_random: run ONLY the current BATTLE screen
    //   using an alternative micro agent, then return. The GameContext is left in
    //   the post-battle state (REWARDS or next screen). Used by LightspeedRunner
    //   when micro_agent_type is 'simple' or 'random'.
    //
    // play_simple_full: run an entire game (combat + overworld) using SimpleAgent's
    //   heuristics. Used as a baseline in agent comparison experiments.

    m.def("play_battle_simple",
        [](GameContext &gc) {
            search::SimpleAgent agent;
            agent.curGameContext = &gc;  // required: playoutBattle reads curGameContext->act
            BattleContext bc;
            bc.init(gc);
            agent.playoutBattle(bc);
            // If maxIter exhausted without resolution, force PLAYER_LOSS so exitBattle
            // sets gc.outcome and prevents the Python runner from looping indefinitely.
            if (bc.outcome == Outcome::UNDECIDED) {
                bc.outcome = Outcome::PLAYER_LOSS;
            }
            bc.exitBattle(gc);
        },
        "Run the current BATTLE using SimpleAgent's deterministic heuristic battle logic."
    );

    m.def("play_battle_random",
        [](GameContext &gc, std::uint32_t seed) {
            std::default_random_engine rng(seed);
            RandomAgent agent(rng);
            BattleContext bc;
            bc.init(gc);
            agent.playoutBattle(bc);
            bc.exitBattle(gc);
        },
        pybind11::arg("gc"),
        pybind11::arg("seed") = 42u,
        "Run the current BATTLE using uniformly random action selection."
    );

    m.def("play_simple_full",
        [](GameContext &gc) {
            search::SimpleAgent agent;
            agent.playout(gc);
        },
        pybind11::arg("gc"),
        "Run a complete game using SimpleAgent's heuristic logic for both combat and overworld."
    );

    pybind11::class_<GameContext> gameContext(m, "GameContext");
    gameContext.def(pybind11::init<CharacterClass, std::uint64_t, int>())
        .def("pick_reward_card", &sts::py::pickRewardCard, "choose to obtain the card at the specified index in the card reward list")
        .def("skip_reward_cards", &sts::py::skipRewardCards, "choose to skip the card reward (increases max_hp by 2 with singing bowl)")
        .def("get_card_reward", &sts::py::getCardReward, "return the current card reward list")
        .def_property_readonly("encounter", [](const GameContext &gc) { return gc.info.encounter; })
        .def_property_readonly("deck",
               [](const GameContext &gc) { return std::vector(gc.deck.cards.begin(), gc.deck.cards.end());},
               "returns a copy of the list of cards in the deck"
        )
        .def("obtain_card",
             [](GameContext &gc, Card card) { gc.deck.obtain(gc, card); },
             "add a card to the deck"
        )
        .def("remove_card",
            [](GameContext &gc, int idx) {
                if (idx < 0 || idx >= gc.deck.size()) {
                    std::cerr << "invalid remove deck remove idx" << std::endl;
                    return;
                }
                gc.deck.remove(gc, idx);
            },
             "remove a card at a idx in the deck"
        )
        .def_property_readonly("relics",
               [] (const GameContext &gc) { return std::vector(gc.relics.relics); },
               "returns a copy of the list of relics"
        )
        .def("__repr__", [](const GameContext &gc) {
            std::ostringstream oss;
            oss << "<" << gc << ">";
            return oss.str();
        }, "returns a string representation of the GameContext");

    // Action methods
    gameContext.def("choose_event_option",
            [](GameContext &gc, int idx) { gc.chooseEventOption(idx); },
            "choose an event option by index")
        .def("choose_boss_relic",
            [](GameContext &gc, int idx) { gc.chooseBossRelic(idx); },
            "choose a boss relic by index")
        .def("choose_campfire_option",
            [](GameContext &gc, int idx) { gc.chooseCampfireOption(idx); },
            "choose a campfire option by index")
        .def("choose_card_select_option",
            [](GameContext &gc, int idx) { gc.chooseSelectCardScreenOption(idx); },
            "choose a card from the card select screen by index")
        .def("choose_treasure_room",
            [](GameContext &gc, bool openChest) { gc.chooseTreasureRoomOption(openChest); },
            "choose whether to open the treasure room chest")
        .def("transition_to_map_node",
            [](GameContext &gc, int x) { gc.transitionToMapNode(x); },
            "transition to the map node at column x")
        .def("choose_neow_option",
            [](GameContext &gc, int idx) { gc.chooseNeowOption(gc.info.neowRewards[idx]); },
            "choose a Neow blessing by index (0-3)")
        .def("shop_buy_card",
            [](GameContext &gc, int idx) { gc.info.shop.buyCard(gc, idx); },
            "buy the card at shop index (0-6)")
        .def("shop_buy_relic",
            [](GameContext &gc, int idx) { gc.info.shop.buyRelic(gc, idx); },
            "buy the relic at shop index (0-2)")
        .def("shop_buy_potion",
            [](GameContext &gc, int idx) { gc.info.shop.buyPotion(gc, idx); },
            "buy the potion at shop index (0-2)")
        .def("shop_purge",
            [](GameContext &gc) { gc.info.shop.buyCardRemove(gc); },
            "pay to purge a card (triggers CARD_SELECT screen)")
        .def("obtain_gold",
            [](GameContext &gc, int amount) { gc.obtainGold(amount); },
            "add gold (for taking gold rewards)")
        .def("obtain_potion",
            [](GameContext &gc, Potion p) { gc.obtainPotion(p); },
            "add a potion to inventory")
        .def("take_reward_gold",
            [](GameContext &gc, int idx) {
                auto &r = gc.info.rewardsContainer;
                if (idx >= 0 && idx < r.goldRewardCount) {
                    gc.obtainGold(r.gold[idx]);
                    r.removeGoldReward(idx);
                }
            },
            "take gold reward at index, add to player gold and remove from rewards container")
        .def("take_reward_relic",
            [](GameContext &gc, int idx) {
                auto &r = gc.info.rewardsContainer;
                if (idx >= 0 && idx < r.relicCount) {
                    gc.obtainRelic(r.relics[idx]);
                    if (r.sapphireKey && idx == r.relicCount - 1) {
                        r.sapphireKey = false;
                    }
                    r.removeRelicReward(idx);
                }
            },
            "take relic reward at index and remove from rewards container")
        .def("skip_reward_relic",
            [](GameContext &gc, int idx) {
                auto &r = gc.info.rewardsContainer;
                if (idx >= 0 && idx < r.relicCount) {
                    r.removeRelicReward(idx);
                }
            },
            "remove relic reward at index without taking it")
        .def("take_reward_key",
            [](GameContext &gc) {
                auto &r = gc.info.rewardsContainer;
                if (r.relicCount > 0) {
                    r.removeRelicReward(r.relicCount - 1);
                }
                gc.obtainKey(Key::SAPPHIRE_KEY);
                r.sapphireKey = false;
            },
            "take sapphire key from reward instead of last relic")
        .def("take_reward_potion",
            [](GameContext &gc, int idx) {
                auto &r = gc.info.rewardsContainer;
                if (idx >= 0 && idx < r.potionCount) {
                    gc.obtainPotion(r.potions[idx]);
                    r.removePotionReward(idx);
                }
            },
            "take potion reward at index and remove from rewards container")
        .def("regain_control",
            [](GameContext &gc) { gc.regainControl(); },
            "advance game state past current screen (e.g. close empty rewards)");

    // State query methods
    gameContext
        .def_property_readonly("cur_event_id",
            [](const GameContext &gc) { return static_cast<int>(gc.curEvent); },
            "current event as integer index into the Event enum")
        .def("get_boss_relics",
            [](const GameContext &gc) {
                return std::vector<RelicId>{
                    gc.info.bossRelics[0], gc.info.bossRelics[1], gc.info.bossRelics[2]
                };
            },
            "return the three boss relic choices")
        .def("get_rewards_info",
            [](const GameContext &gc) -> pybind11::dict {
                const auto &r = gc.info.rewardsContainer;
                pybind11::dict d;
                d["gold_count"]        = r.goldRewardCount;
                d["gold"]              = std::vector<int>(r.gold.begin(), r.gold.begin() + r.goldRewardCount);
                d["relic_count"]       = r.relicCount;
                d["relics"]            = std::vector<RelicId>(r.relics.begin(), r.relics.begin() + r.relicCount);
                d["potion_count"]      = r.potionCount;
                d["potions"]           = std::vector<Potion>(r.potions.begin(), r.potions.begin() + r.potionCount);
                d["card_reward_count"] = r.cardRewardCount;
                d["emerald_key"]       = r.emeraldKey;
                d["sapphire_key"]      = r.sapphireKey;
                return d;
            },
            "return the current rewards container as a dict")
        .def("get_shop_info",
            [](const GameContext &gc) -> pybind11::dict {
                const auto &s = gc.info.shop;
                pybind11::dict d;
                // cards[0-6], relics[0-2], potions[0-2]
                // prices layout: [0-6]=cards, [7-9]=relics, [10-12]=potions
                std::vector<Card> cards(s.cards, s.cards + 7);
                std::vector<RelicId> relics(s.relics, s.relics + 3);
                std::vector<Potion> potions(s.potions, s.potions + 3);
                std::vector<int> card_prices, relic_prices, potion_prices;
                for (int i = 0; i < 7; ++i) card_prices.push_back(s.prices[i]);
                for (int i = 7; i < 10; ++i) relic_prices.push_back(s.prices[i]);
                for (int i = 10; i < 13; ++i) potion_prices.push_back(s.prices[i]);
                d["cards"]         = cards;
                d["card_prices"]   = card_prices;
                d["relics"]        = relics;
                d["relic_prices"]  = relic_prices;
                d["potions"]       = potions;
                d["potion_prices"] = potion_prices;
                d["remove_cost"]   = s.removeCost;
                return d;
            },
            "return current shop inventory as a dict")
        .def("get_card_select_info",
            [](const GameContext &gc) -> pybind11::dict {
                pybind11::dict d;
                d["screen_type"]     = static_cast<int>(gc.info.selectScreenType);
                d["to_select_count"] = gc.info.toSelectCount;
                pybind11::list cards;
                for (int i = 0; i < (int)gc.info.toSelectCards.size(); ++i) {
                    const auto &sc = gc.info.toSelectCards[i];
                    cards.append(pybind11::make_tuple(sc.card, sc.deckIdx));
                }
                d["cards"] = cards;
                return d;
            },
            "return card select screen info as a dict")
        .def("get_neow_rewards",
            [](const GameContext &gc) -> pybind11::list {
                pybind11::list rewards;
                for (int i = 0; i < 4; ++i) {
                    const auto &nr = gc.info.neowRewards[i];
                    pybind11::dict d;
                    d["bonus"] = static_cast<int>(nr.r);
                    d["cost"]  = static_cast<int>(nr.d);
                    rewards.append(d);
                }
                return rewards;
            },
            "return the four Neow blessing options as a list of dicts with 'bonus' and 'cost' ints")
        .def("get_potions",
            [](const GameContext &gc) -> std::vector<Potion> {
                return std::vector<Potion>(gc.potions.begin(), gc.potions.begin() + gc.potionCount);
            },
            "return current potion inventory (occupied slots only)");

    gameContext.def_readwrite("outcome", &GameContext::outcome)
        .def_readwrite("act", &GameContext::act)
        .def_readwrite("floor_num", &GameContext::floorNum)
        .def_readwrite("screen_state", &GameContext::screenState)

        .def_readwrite("seed", &GameContext::seed)
        .def_readwrite("cur_map_node_x", &GameContext::curMapNodeX)
        .def_readwrite("cur_map_node_y", &GameContext::curMapNodeY)
        .def_readwrite("cur_room", &GameContext::curRoom)
        .def_readwrite("boss", &GameContext::boss)

        .def_readwrite("cur_hp", &GameContext::curHp)
        .def_readwrite("max_hp", &GameContext::maxHp)
        .def_readwrite("gold", &GameContext::gold)

        .def_readwrite("blue_key", &GameContext::blueKey)
        .def_readwrite("green_key", &GameContext::greenKey)
        .def_readwrite("red_key", &GameContext::redKey)

        .def_readwrite("card_rarity_factor", &GameContext::cardRarityFactor)
        .def_readwrite("potion_chance", &GameContext::potionChance)
        .def_readwrite("monster_chance", &GameContext::monsterChance)
        .def_readwrite("shop_chance", &GameContext::shopChance)
        .def_readwrite("treasure_chance", &GameContext::treasureChance)

        .def_readwrite("shop_remove_count", &GameContext::shopRemoveCount)
        .def_readwrite("speedrun_pace", &GameContext::speedrunPace)
        .def_readwrite("note_for_yourself_card", &GameContext::noteForYourselfCard)
        .def("get_bottled_card_indices",
            [](const GameContext &gc) -> std::vector<int> {
                std::vector<int> result;
                for (int i = 0; i < gc.deck.size(); ++i) {
                    if (gc.deck.isCardBottled(i)) {
                        result.push_back(i);
                    }
                }
                return result;
            },
            "return list of deck indices that are bottled (excluded from REMOVE/TOKE selection)");

    pybind11::class_<RelicInstance> relic(m, "Relic");
    relic.def_readwrite("id", &RelicInstance::id)
        .def_readwrite("data", &RelicInstance::data);

    pybind11::class_<Map> map(m, "SpireMap");
    map.def(pybind11::init<std::uint64_t, int, int, bool>(),
            pybind11::arg("seed"), pybind11::arg("ascension"), pybind11::arg("act"), pybind11::arg("assign_burning_elite"));
    map.def("get_room_type", &sts::py::getRoomType);
    map.def("has_edge", &sts::py::hasEdge);
    map.def("get_nn_rep", &sts::py::getNNMapRepresentation);
    map.def_readonly("burning_elite_x", &Map::burningEliteX);
    map.def_readonly("burning_elite_y", &Map::burningEliteY);
    map.def_readonly("burning_elite_buff", &Map::burningEliteBuff);
    map.def("__repr__", [](const Map &m) {
        return m.toString(true);
    });

    pybind11::class_<Card> card(m, "Card");
    card.def(pybind11::init<CardId>())
        .def("__repr__", [](const Card &c) {
            std::string s("<slaythespire.Card ");
            s += c.getName();
            if (c.isUpgraded()) {
                s += '+';
                if (c.id == sts::CardId::SEARING_BLOW) {
                    s += std::to_string(c.getUpgraded());
                }
            }
            return s += ">";
        }, "returns a string representation of a Card")
        .def("upgrade", &Card::upgrade)
        .def_readwrite("misc", &Card::misc, "value internal to the simulator used for things like ritual dagger damage");

    card.def_property_readonly("id", &Card::getId)
        .def_property_readonly("upgraded", &Card::isUpgraded)
        .def_property_readonly("upgrade_count", &Card::getUpgraded)
        .def_property_readonly("innate", &Card::isInnate)
        .def_property_readonly("transformable", &Card::canTransform)
        .def_property_readonly("upgradable", &Card::canUpgrade)
        .def_property_readonly("is_strikeCard", &Card::isStrikeCard)
        .def_property_readonly("is_starter_strike_or_defend", &Card::isStarterStrikeOrDefend)
        .def_property_readonly("rarity", &Card::getRarity)
        .def_property_readonly("type", &Card::getType);

    pybind11::enum_<GameOutcome> gameOutcome(m, "GameOutcome");
    gameOutcome.value("UNDECIDED", GameOutcome::UNDECIDED)
        .value("PLAYER_VICTORY", GameOutcome::PLAYER_VICTORY)
        .value("PLAYER_LOSS", GameOutcome::PLAYER_LOSS);

    pybind11::enum_<ScreenState> screenState(m, "ScreenState");
    screenState.value("INVALID", ScreenState::INVALID)
        .value("EVENT_SCREEN", ScreenState::EVENT_SCREEN)
        .value("REWARDS", ScreenState::REWARDS)
        .value("BOSS_RELIC_REWARDS", ScreenState::BOSS_RELIC_REWARDS)
        .value("CARD_SELECT", ScreenState::CARD_SELECT)
        .value("MAP_SCREEN", ScreenState::MAP_SCREEN)
        .value("TREASURE_ROOM", ScreenState::TREASURE_ROOM)
        .value("REST_ROOM", ScreenState::REST_ROOM)
        .value("SHOP_ROOM", ScreenState::SHOP_ROOM)
        .value("BATTLE", ScreenState::BATTLE);

    pybind11::enum_<CharacterClass> characterClass(m, "CharacterClass");
    characterClass.value("IRONCLAD", CharacterClass::IRONCLAD)
            .value("SILENT", CharacterClass::SILENT)
            .value("DEFECT", CharacterClass::DEFECT)
            .value("WATCHER", CharacterClass::WATCHER)
            .value("INVALID", CharacterClass::INVALID);

    pybind11::enum_<Room> roomEnum(m, "Room");
    roomEnum.value("SHOP", Room::SHOP)
        .value("REST", Room::REST)
        .value("EVENT", Room::EVENT)
        .value("ELITE", Room::ELITE)
        .value("MONSTER", Room::MONSTER)
        .value("TREASURE", Room::TREASURE)
        .value("BOSS", Room::BOSS)
        .value("BOSS_TREASURE", Room::BOSS_TREASURE)
        .value("NONE", Room::NONE)
        .value("INVALID", Room::INVALID);

    pybind11::enum_<CardRarity>(m, "CardRarity")
        .value("COMMON", CardRarity::COMMON)
        .value("UNCOMMON", CardRarity::UNCOMMON)
        .value("RARE", CardRarity::RARE)
        .value("BASIC", CardRarity::BASIC)
        .value("SPECIAL", CardRarity::SPECIAL)
        .value("CURSE", CardRarity::CURSE)
        .value("INVALID", CardRarity::INVALID);

    pybind11::enum_<CardColor>(m, "CardColor")
        .value("RED", CardColor::RED)
        .value("GREEN", CardColor::GREEN)
        .value("PURPLE", CardColor::PURPLE)
        .value("COLORLESS", CardColor::COLORLESS)
        .value("CURSE", CardColor::CURSE)
        .value("INVALID", CardColor::INVALID);

    pybind11::enum_<CardType>(m, "CardType")
        .value("ATTACK", CardType::ATTACK)
        .value("SKILL", CardType::SKILL)
        .value("POWER", CardType::POWER)
        .value("CURSE", CardType::CURSE)
        .value("STATUS", CardType::STATUS)
        .value("INVALID", CardType::INVALID);

    pybind11::enum_<CardId>(m, "CardId")
        .value("INVALID", CardId::INVALID)
        .value("ACCURACY", CardId::ACCURACY)
        .value("ACROBATICS", CardId::ACROBATICS)
        .value("ADRENALINE", CardId::ADRENALINE)
        .value("AFTER_IMAGE", CardId::AFTER_IMAGE)
        .value("AGGREGATE", CardId::AGGREGATE)
        .value("ALCHEMIZE", CardId::ALCHEMIZE)
        .value("ALL_FOR_ONE", CardId::ALL_FOR_ONE)
        .value("ALL_OUT_ATTACK", CardId::ALL_OUT_ATTACK)
        .value("ALPHA", CardId::ALPHA)
        .value("AMPLIFY", CardId::AMPLIFY)
        .value("ANGER", CardId::ANGER)
        .value("APOTHEOSIS", CardId::APOTHEOSIS)
        .value("APPARITION", CardId::APPARITION)
        .value("ARMAMENTS", CardId::ARMAMENTS)
        .value("ASCENDERS_BANE", CardId::ASCENDERS_BANE)
        .value("AUTO_SHIELDS", CardId::AUTO_SHIELDS)
        .value("A_THOUSAND_CUTS", CardId::A_THOUSAND_CUTS)
        .value("BACKFLIP", CardId::BACKFLIP)
        .value("BACKSTAB", CardId::BACKSTAB)
        .value("BALL_LIGHTNING", CardId::BALL_LIGHTNING)
        .value("BANDAGE_UP", CardId::BANDAGE_UP)
        .value("BANE", CardId::BANE)
        .value("BARRAGE", CardId::BARRAGE)
        .value("BARRICADE", CardId::BARRICADE)
        .value("BASH", CardId::BASH)
        .value("BATTLE_HYMN", CardId::BATTLE_HYMN)
        .value("BATTLE_TRANCE", CardId::BATTLE_TRANCE)
        .value("BEAM_CELL", CardId::BEAM_CELL)
        .value("BECOME_ALMIGHTY", CardId::BECOME_ALMIGHTY)
        .value("BERSERK", CardId::BERSERK)
        .value("BETA", CardId::BETA)
        .value("BIASED_COGNITION", CardId::BIASED_COGNITION)
        .value("BITE", CardId::BITE)
        .value("BLADE_DANCE", CardId::BLADE_DANCE)
        .value("BLASPHEMY", CardId::BLASPHEMY)
        .value("BLIND", CardId::BLIND)
        .value("BLIZZARD", CardId::BLIZZARD)
        .value("BLOODLETTING", CardId::BLOODLETTING)
        .value("BLOOD_FOR_BLOOD", CardId::BLOOD_FOR_BLOOD)
        .value("BLUDGEON", CardId::BLUDGEON)
        .value("BLUR", CardId::BLUR)
        .value("BODY_SLAM", CardId::BODY_SLAM)
        .value("BOOT_SEQUENCE", CardId::BOOT_SEQUENCE)
        .value("BOUNCING_FLASK", CardId::BOUNCING_FLASK)
        .value("BOWLING_BASH", CardId::BOWLING_BASH)
        .value("BRILLIANCE", CardId::BRILLIANCE)
        .value("BRUTALITY", CardId::BRUTALITY)
        .value("BUFFER", CardId::BUFFER)
        .value("BULLET_TIME", CardId::BULLET_TIME)
        .value("BULLSEYE", CardId::BULLSEYE)
        .value("BURN", CardId::BURN)
        .value("BURNING_PACT", CardId::BURNING_PACT)
        .value("BURST", CardId::BURST)
        .value("CALCULATED_GAMBLE", CardId::CALCULATED_GAMBLE)
        .value("CALTROPS", CardId::CALTROPS)
        .value("CAPACITOR", CardId::CAPACITOR)
        .value("CARNAGE", CardId::CARNAGE)
        .value("CARVE_REALITY", CardId::CARVE_REALITY)
        .value("CATALYST", CardId::CATALYST)
        .value("CHAOS", CardId::CHAOS)
        .value("CHARGE_BATTERY", CardId::CHARGE_BATTERY)
        .value("CHILL", CardId::CHILL)
        .value("CHOKE", CardId::CHOKE)
        .value("CHRYSALIS", CardId::CHRYSALIS)
        .value("CLASH", CardId::CLASH)
        .value("CLAW", CardId::CLAW)
        .value("CLEAVE", CardId::CLEAVE)
        .value("CLOAK_AND_DAGGER", CardId::CLOAK_AND_DAGGER)
        .value("CLOTHESLINE", CardId::CLOTHESLINE)
        .value("CLUMSY", CardId::CLUMSY)
        .value("COLD_SNAP", CardId::COLD_SNAP)
        .value("COLLECT", CardId::COLLECT)
        .value("COMBUST", CardId::COMBUST)
        .value("COMPILE_DRIVER", CardId::COMPILE_DRIVER)
        .value("CONCENTRATE", CardId::CONCENTRATE)
        .value("CONCLUDE", CardId::CONCLUDE)
        .value("CONJURE_BLADE", CardId::CONJURE_BLADE)
        .value("CONSECRATE", CardId::CONSECRATE)
        .value("CONSUME", CardId::CONSUME)
        .value("COOLHEADED", CardId::COOLHEADED)
        .value("CORE_SURGE", CardId::CORE_SURGE)
        .value("CORPSE_EXPLOSION", CardId::CORPSE_EXPLOSION)
        .value("CORRUPTION", CardId::CORRUPTION)
        .value("CREATIVE_AI", CardId::CREATIVE_AI)
        .value("CRESCENDO", CardId::CRESCENDO)
        .value("CRIPPLING_CLOUD", CardId::CRIPPLING_CLOUD)
        .value("CRUSH_JOINTS", CardId::CRUSH_JOINTS)
        .value("CURSE_OF_THE_BELL", CardId::CURSE_OF_THE_BELL)
        .value("CUT_THROUGH_FATE", CardId::CUT_THROUGH_FATE)
        .value("DAGGER_SPRAY", CardId::DAGGER_SPRAY)
        .value("DAGGER_THROW", CardId::DAGGER_THROW)
        .value("DARKNESS", CardId::DARKNESS)
        .value("DARK_EMBRACE", CardId::DARK_EMBRACE)
        .value("DARK_SHACKLES", CardId::DARK_SHACKLES)
        .value("DASH", CardId::DASH)
        .value("DAZED", CardId::DAZED)
        .value("DEADLY_POISON", CardId::DEADLY_POISON)
        .value("DECAY", CardId::DECAY)
        .value("DECEIVE_REALITY", CardId::DECEIVE_REALITY)
        .value("DEEP_BREATH", CardId::DEEP_BREATH)
        .value("DEFEND_BLUE", CardId::DEFEND_BLUE)
        .value("DEFEND_GREEN", CardId::DEFEND_GREEN)
        .value("DEFEND_PURPLE", CardId::DEFEND_PURPLE)
        .value("DEFEND_RED", CardId::DEFEND_RED)
        .value("DEFLECT", CardId::DEFLECT)
        .value("DEFRAGMENT", CardId::DEFRAGMENT)
        .value("DEMON_FORM", CardId::DEMON_FORM)
        .value("DEUS_EX_MACHINA", CardId::DEUS_EX_MACHINA)
        .value("DEVA_FORM", CardId::DEVA_FORM)
        .value("DEVOTION", CardId::DEVOTION)
        .value("DIE_DIE_DIE", CardId::DIE_DIE_DIE)
        .value("DISARM", CardId::DISARM)
        .value("DISCOVERY", CardId::DISCOVERY)
        .value("DISTRACTION", CardId::DISTRACTION)
        .value("DODGE_AND_ROLL", CardId::DODGE_AND_ROLL)
        .value("DOOM_AND_GLOOM", CardId::DOOM_AND_GLOOM)
        .value("DOPPELGANGER", CardId::DOPPELGANGER)
        .value("DOUBLE_ENERGY", CardId::DOUBLE_ENERGY)
        .value("DOUBLE_TAP", CardId::DOUBLE_TAP)
        .value("DOUBT", CardId::DOUBT)
        .value("DRAMATIC_ENTRANCE", CardId::DRAMATIC_ENTRANCE)
        .value("DROPKICK", CardId::DROPKICK)
        .value("DUALCAST", CardId::DUALCAST)
        .value("DUAL_WIELD", CardId::DUAL_WIELD)
        .value("ECHO_FORM", CardId::ECHO_FORM)
        .value("ELECTRODYNAMICS", CardId::ELECTRODYNAMICS)
        .value("EMPTY_BODY", CardId::EMPTY_BODY)
        .value("EMPTY_FIST", CardId::EMPTY_FIST)
        .value("EMPTY_MIND", CardId::EMPTY_MIND)
        .value("ENDLESS_AGONY", CardId::ENDLESS_AGONY)
        .value("ENLIGHTENMENT", CardId::ENLIGHTENMENT)
        .value("ENTRENCH", CardId::ENTRENCH)
        .value("ENVENOM", CardId::ENVENOM)
        .value("EQUILIBRIUM", CardId::EQUILIBRIUM)
        .value("ERUPTION", CardId::ERUPTION)
        .value("ESCAPE_PLAN", CardId::ESCAPE_PLAN)
        .value("ESTABLISHMENT", CardId::ESTABLISHMENT)
        .value("EVALUATE", CardId::EVALUATE)
        .value("EVISCERATE", CardId::EVISCERATE)
        .value("EVOLVE", CardId::EVOLVE)
        .value("EXHUME", CardId::EXHUME)
        .value("EXPERTISE", CardId::EXPERTISE)
        .value("EXPUNGER", CardId::EXPUNGER)
        .value("FAME_AND_FORTUNE", CardId::FAME_AND_FORTUNE)
        .value("FASTING", CardId::FASTING)
        .value("FEAR_NO_EVIL", CardId::FEAR_NO_EVIL)
        .value("FEED", CardId::FEED)
        .value("FEEL_NO_PAIN", CardId::FEEL_NO_PAIN)
        .value("FIEND_FIRE", CardId::FIEND_FIRE)
        .value("FINESSE", CardId::FINESSE)
        .value("FINISHER", CardId::FINISHER)
        .value("FIRE_BREATHING", CardId::FIRE_BREATHING)
        .value("FISSION", CardId::FISSION)
        .value("FLAME_BARRIER", CardId::FLAME_BARRIER)
        .value("FLASH_OF_STEEL", CardId::FLASH_OF_STEEL)
        .value("FLECHETTES", CardId::FLECHETTES)
        .value("FLEX", CardId::FLEX)
        .value("FLURRY_OF_BLOWS", CardId::FLURRY_OF_BLOWS)
        .value("FLYING_KNEE", CardId::FLYING_KNEE)
        .value("FLYING_SLEEVES", CardId::FLYING_SLEEVES)
        .value("FOLLOW_UP", CardId::FOLLOW_UP)
        .value("FOOTWORK", CardId::FOOTWORK)
        .value("FORCE_FIELD", CardId::FORCE_FIELD)
        .value("FOREIGN_INFLUENCE", CardId::FOREIGN_INFLUENCE)
        .value("FORESIGHT", CardId::FORESIGHT)
        .value("FORETHOUGHT", CardId::FORETHOUGHT)
        .value("FTL", CardId::FTL)
        .value("FUSION", CardId::FUSION)
        .value("GENETIC_ALGORITHM", CardId::GENETIC_ALGORITHM)
        .value("GHOSTLY_ARMOR", CardId::GHOSTLY_ARMOR)
        .value("GLACIER", CardId::GLACIER)
        .value("GLASS_KNIFE", CardId::GLASS_KNIFE)
        .value("GOOD_INSTINCTS", CardId::GOOD_INSTINCTS)
        .value("GO_FOR_THE_EYES", CardId::GO_FOR_THE_EYES)
        .value("GRAND_FINALE", CardId::GRAND_FINALE)
        .value("HALT", CardId::HALT)
        .value("HAND_OF_GREED", CardId::HAND_OF_GREED)
        .value("HAVOC", CardId::HAVOC)
        .value("HEADBUTT", CardId::HEADBUTT)
        .value("HEATSINKS", CardId::HEATSINKS)
        .value("HEAVY_BLADE", CardId::HEAVY_BLADE)
        .value("HEEL_HOOK", CardId::HEEL_HOOK)
        .value("HELLO_WORLD", CardId::HELLO_WORLD)
        .value("HEMOKINESIS", CardId::HEMOKINESIS)
        .value("HOLOGRAM", CardId::HOLOGRAM)
        .value("HYPERBEAM", CardId::HYPERBEAM)
        .value("IMMOLATE", CardId::IMMOLATE)
        .value("IMPATIENCE", CardId::IMPATIENCE)
        .value("IMPERVIOUS", CardId::IMPERVIOUS)
        .value("INDIGNATION", CardId::INDIGNATION)
        .value("INFERNAL_BLADE", CardId::INFERNAL_BLADE)
        .value("INFINITE_BLADES", CardId::INFINITE_BLADES)
        .value("INFLAME", CardId::INFLAME)
        .value("INJURY", CardId::INJURY)
        .value("INNER_PEACE", CardId::INNER_PEACE)
        .value("INSIGHT", CardId::INSIGHT)
        .value("INTIMIDATE", CardId::INTIMIDATE)
        .value("IRON_WAVE", CardId::IRON_WAVE)
        .value("JAX", CardId::JAX)
        .value("JACK_OF_ALL_TRADES", CardId::JACK_OF_ALL_TRADES)
        .value("JUDGMENT", CardId::JUDGMENT)
        .value("JUGGERNAUT", CardId::JUGGERNAUT)
        .value("JUST_LUCKY", CardId::JUST_LUCKY)
        .value("LEAP", CardId::LEAP)
        .value("LEG_SWEEP", CardId::LEG_SWEEP)
        .value("LESSON_LEARNED", CardId::LESSON_LEARNED)
        .value("LIKE_WATER", CardId::LIKE_WATER)
        .value("LIMIT_BREAK", CardId::LIMIT_BREAK)
        .value("LIVE_FOREVER", CardId::LIVE_FOREVER)
        .value("LOOP", CardId::LOOP)
        .value("MACHINE_LEARNING", CardId::MACHINE_LEARNING)
        .value("MADNESS", CardId::MADNESS)
        .value("MAGNETISM", CardId::MAGNETISM)
        .value("MALAISE", CardId::MALAISE)
        .value("MASTERFUL_STAB", CardId::MASTERFUL_STAB)
        .value("MASTER_OF_STRATEGY", CardId::MASTER_OF_STRATEGY)
        .value("MASTER_REALITY", CardId::MASTER_REALITY)
        .value("MAYHEM", CardId::MAYHEM)
        .value("MEDITATE", CardId::MEDITATE)
        .value("MELTER", CardId::MELTER)
        .value("MENTAL_FORTRESS", CardId::MENTAL_FORTRESS)
        .value("METALLICIZE", CardId::METALLICIZE)
        .value("METAMORPHOSIS", CardId::METAMORPHOSIS)
        .value("METEOR_STRIKE", CardId::METEOR_STRIKE)
        .value("MIND_BLAST", CardId::MIND_BLAST)
        .value("MIRACLE", CardId::MIRACLE)
        .value("MULTI_CAST", CardId::MULTI_CAST)
        .value("NECRONOMICURSE", CardId::NECRONOMICURSE)
        .value("NEUTRALIZE", CardId::NEUTRALIZE)
        .value("NIGHTMARE", CardId::NIGHTMARE)
        .value("NIRVANA", CardId::NIRVANA)
        .value("NORMALITY", CardId::NORMALITY)
        .value("NOXIOUS_FUMES", CardId::NOXIOUS_FUMES)
        .value("OFFERING", CardId::OFFERING)
        .value("OMEGA", CardId::OMEGA)
        .value("OMNISCIENCE", CardId::OMNISCIENCE)
        .value("OUTMANEUVER", CardId::OUTMANEUVER)
        .value("OVERCLOCK", CardId::OVERCLOCK)
        .value("PAIN", CardId::PAIN)
        .value("PANACEA", CardId::PANACEA)
        .value("PANACHE", CardId::PANACHE)
        .value("PANIC_BUTTON", CardId::PANIC_BUTTON)
        .value("PARASITE", CardId::PARASITE)
        .value("PERFECTED_STRIKE", CardId::PERFECTED_STRIKE)
        .value("PERSEVERANCE", CardId::PERSEVERANCE)
        .value("PHANTASMAL_KILLER", CardId::PHANTASMAL_KILLER)
        .value("PIERCING_WAIL", CardId::PIERCING_WAIL)
        .value("POISONED_STAB", CardId::POISONED_STAB)
        .value("POMMEL_STRIKE", CardId::POMMEL_STRIKE)
        .value("POWER_THROUGH", CardId::POWER_THROUGH)
        .value("PRAY", CardId::PRAY)
        .value("PREDATOR", CardId::PREDATOR)
        .value("PREPARED", CardId::PREPARED)
        .value("PRESSURE_POINTS", CardId::PRESSURE_POINTS)
        .value("PRIDE", CardId::PRIDE)
        .value("PROSTRATE", CardId::PROSTRATE)
        .value("PROTECT", CardId::PROTECT)
        .value("PUMMEL", CardId::PUMMEL)
        .value("PURITY", CardId::PURITY)
        .value("QUICK_SLASH", CardId::QUICK_SLASH)
        .value("RAGE", CardId::RAGE)
        .value("RAGNAROK", CardId::RAGNAROK)
        .value("RAINBOW", CardId::RAINBOW)
        .value("RAMPAGE", CardId::RAMPAGE)
        .value("REACH_HEAVEN", CardId::REACH_HEAVEN)
        .value("REAPER", CardId::REAPER)
        .value("REBOOT", CardId::REBOOT)
        .value("REBOUND", CardId::REBOUND)
        .value("RECKLESS_CHARGE", CardId::RECKLESS_CHARGE)
        .value("RECURSION", CardId::RECURSION)
        .value("RECYCLE", CardId::RECYCLE)
        .value("REFLEX", CardId::REFLEX)
        .value("REGRET", CardId::REGRET)
        .value("REINFORCED_BODY", CardId::REINFORCED_BODY)
        .value("REPROGRAM", CardId::REPROGRAM)
        .value("RIDDLE_WITH_HOLES", CardId::RIDDLE_WITH_HOLES)
        .value("RIP_AND_TEAR", CardId::RIP_AND_TEAR)
        .value("RITUAL_DAGGER", CardId::RITUAL_DAGGER)
        .value("RUPTURE", CardId::RUPTURE)
        .value("RUSHDOWN", CardId::RUSHDOWN)
        .value("SADISTIC_NATURE", CardId::SADISTIC_NATURE)
        .value("SAFETY", CardId::SAFETY)
        .value("SANCTITY", CardId::SANCTITY)
        .value("SANDS_OF_TIME", CardId::SANDS_OF_TIME)
        .value("SASH_WHIP", CardId::SASH_WHIP)
        .value("SCRAPE", CardId::SCRAPE)
        .value("SCRAWL", CardId::SCRAWL)
        .value("SEARING_BLOW", CardId::SEARING_BLOW)
        .value("SECOND_WIND", CardId::SECOND_WIND)
        .value("SECRET_TECHNIQUE", CardId::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardId::SECRET_WEAPON)
        .value("SEEING_RED", CardId::SEEING_RED)
        .value("SEEK", CardId::SEEK)
        .value("SELF_REPAIR", CardId::SELF_REPAIR)
        .value("SENTINEL", CardId::SENTINEL)
        .value("SETUP", CardId::SETUP)
        .value("SEVER_SOUL", CardId::SEVER_SOUL)
        .value("SHAME", CardId::SHAME)
        .value("SHIV", CardId::SHIV)
        .value("SHOCKWAVE", CardId::SHOCKWAVE)
        .value("SHRUG_IT_OFF", CardId::SHRUG_IT_OFF)
        .value("SIGNATURE_MOVE", CardId::SIGNATURE_MOVE)
        .value("SIMMERING_FURY", CardId::SIMMERING_FURY)
        .value("SKEWER", CardId::SKEWER)
        .value("SKIM", CardId::SKIM)
        .value("SLICE", CardId::SLICE)
        .value("SLIMED", CardId::SLIMED)
        .value("SMITE", CardId::SMITE)
        .value("SNEAKY_STRIKE", CardId::SNEAKY_STRIKE)
        .value("SPIRIT_SHIELD", CardId::SPIRIT_SHIELD)
        .value("SPOT_WEAKNESS", CardId::SPOT_WEAKNESS)
        .value("STACK", CardId::STACK)
        .value("STATIC_DISCHARGE", CardId::STATIC_DISCHARGE)
        .value("STEAM_BARRIER", CardId::STEAM_BARRIER)
        .value("STORM", CardId::STORM)
        .value("STORM_OF_STEEL", CardId::STORM_OF_STEEL)
        .value("STREAMLINE", CardId::STREAMLINE)
        .value("STRIKE_BLUE", CardId::STRIKE_BLUE)
        .value("STRIKE_GREEN", CardId::STRIKE_GREEN)
        .value("STRIKE_PURPLE", CardId::STRIKE_PURPLE)
        .value("STRIKE_RED", CardId::STRIKE_RED)
        .value("STUDY", CardId::STUDY)
        .value("SUCKER_PUNCH", CardId::SUCKER_PUNCH)
        .value("SUNDER", CardId::SUNDER)
        .value("SURVIVOR", CardId::SURVIVOR)
        .value("SWEEPING_BEAM", CardId::SWEEPING_BEAM)
        .value("SWIFT_STRIKE", CardId::SWIFT_STRIKE)
        .value("SWIVEL", CardId::SWIVEL)
        .value("SWORD_BOOMERANG", CardId::SWORD_BOOMERANG)
        .value("TACTICIAN", CardId::TACTICIAN)
        .value("TALK_TO_THE_HAND", CardId::TALK_TO_THE_HAND)
        .value("TANTRUM", CardId::TANTRUM)
        .value("TEMPEST", CardId::TEMPEST)
        .value("TERROR", CardId::TERROR)
        .value("THE_BOMB", CardId::THE_BOMB)
        .value("THINKING_AHEAD", CardId::THINKING_AHEAD)
        .value("THIRD_EYE", CardId::THIRD_EYE)
        .value("THROUGH_VIOLENCE", CardId::THROUGH_VIOLENCE)
        .value("THUNDERCLAP", CardId::THUNDERCLAP)
        .value("THUNDER_STRIKE", CardId::THUNDER_STRIKE)
        .value("TOOLS_OF_THE_TRADE", CardId::TOOLS_OF_THE_TRADE)
        .value("TRANQUILITY", CardId::TRANQUILITY)
        .value("TRANSMUTATION", CardId::TRANSMUTATION)
        .value("TRIP", CardId::TRIP)
        .value("TRUE_GRIT", CardId::TRUE_GRIT)
        .value("TURBO", CardId::TURBO)
        .value("TWIN_STRIKE", CardId::TWIN_STRIKE)
        .value("UNLOAD", CardId::UNLOAD)
        .value("UPPERCUT", CardId::UPPERCUT)
        .value("VAULT", CardId::VAULT)
        .value("VIGILANCE", CardId::VIGILANCE)
        .value("VIOLENCE", CardId::VIOLENCE)
        .value("VOID", CardId::VOID)
        .value("WALLOP", CardId::WALLOP)
        .value("WARCRY", CardId::WARCRY)
        .value("WAVE_OF_THE_HAND", CardId::WAVE_OF_THE_HAND)
        .value("WEAVE", CardId::WEAVE)
        .value("WELL_LAID_PLANS", CardId::WELL_LAID_PLANS)
        .value("WHEEL_KICK", CardId::WHEEL_KICK)
        .value("WHIRLWIND", CardId::WHIRLWIND)
        .value("WHITE_NOISE", CardId::WHITE_NOISE)
        .value("WILD_STRIKE", CardId::WILD_STRIKE)
        .value("WINDMILL_STRIKE", CardId::WINDMILL_STRIKE)
        .value("WISH", CardId::WISH)
        .value("WORSHIP", CardId::WORSHIP)
        .value("WOUND", CardId::WOUND)
        .value("WRAITH_FORM", CardId::WRAITH_FORM)
        .value("WREATH_OF_FLAME", CardId::WREATH_OF_FLAME)
        .value("WRITHE", CardId::WRITHE)
        .value("ZAP", CardId::ZAP);

    pybind11::enum_<MonsterEncounter> meEnum(m, "MonsterEncounter");
    meEnum.value("INVALID", ME::INVALID)
        .value("CULTIST", ME::CULTIST)
        .value("JAW_WORM", ME::JAW_WORM)
        .value("TWO_LOUSE", ME::TWO_LOUSE)
        .value("SMALL_SLIMES", ME::SMALL_SLIMES)
        .value("BLUE_SLAVER", ME::BLUE_SLAVER)
        .value("GREMLIN_GANG", ME::GREMLIN_GANG)
        .value("LOOTER", ME::LOOTER)
        .value("LARGE_SLIME", ME::LARGE_SLIME)
        .value("LOTS_OF_SLIMES", ME::LOTS_OF_SLIMES)
        .value("EXORDIUM_THUGS", ME::EXORDIUM_THUGS)
        .value("EXORDIUM_WILDLIFE", ME::EXORDIUM_WILDLIFE)
        .value("RED_SLAVER", ME::RED_SLAVER)
        .value("THREE_LOUSE", ME::THREE_LOUSE)
        .value("TWO_FUNGI_BEASTS", ME::TWO_FUNGI_BEASTS)
        .value("GREMLIN_NOB", ME::GREMLIN_NOB)
        .value("LAGAVULIN", ME::LAGAVULIN)
        .value("THREE_SENTRIES", ME::THREE_SENTRIES)
        .value("SLIME_BOSS", ME::SLIME_BOSS)
        .value("THE_GUARDIAN", ME::THE_GUARDIAN)
        .value("HEXAGHOST", ME::HEXAGHOST)
        .value("SPHERIC_GUARDIAN", ME::SPHERIC_GUARDIAN)
        .value("CHOSEN", ME::CHOSEN)
        .value("SHELL_PARASITE", ME::SHELL_PARASITE)
        .value("THREE_BYRDS", ME::THREE_BYRDS)
        .value("TWO_THIEVES", ME::TWO_THIEVES)
        .value("CHOSEN_AND_BYRDS", ME::CHOSEN_AND_BYRDS)
        .value("SENTRY_AND_SPHERE", ME::SENTRY_AND_SPHERE)
        .value("SNAKE_PLANT", ME::SNAKE_PLANT)
        .value("SNECKO", ME::SNECKO)
        .value("CENTURION_AND_HEALER", ME::CENTURION_AND_HEALER)
        .value("CULTIST_AND_CHOSEN", ME::CULTIST_AND_CHOSEN)
        .value("THREE_CULTIST", ME::THREE_CULTIST)
        .value("SHELLED_PARASITE_AND_FUNGI", ME::SHELLED_PARASITE_AND_FUNGI)
        .value("GREMLIN_LEADER", ME::GREMLIN_LEADER)
        .value("SLAVERS", ME::SLAVERS)
        .value("BOOK_OF_STABBING", ME::BOOK_OF_STABBING)
        .value("AUTOMATON", ME::AUTOMATON)
        .value("COLLECTOR", ME::COLLECTOR)
        .value("CHAMP", ME::CHAMP)
        .value("THREE_DARKLINGS", ME::THREE_DARKLINGS)
        .value("ORB_WALKER", ME::ORB_WALKER)
        .value("THREE_SHAPES", ME::THREE_SHAPES)
        .value("SPIRE_GROWTH", ME::SPIRE_GROWTH)
        .value("TRANSIENT", ME::TRANSIENT)
        .value("FOUR_SHAPES", ME::FOUR_SHAPES)
        .value("MAW", ME::MAW)
        .value("SPHERE_AND_TWO_SHAPES", ME::SPHERE_AND_TWO_SHAPES)
        .value("JAW_WORM_HORDE", ME::JAW_WORM_HORDE)
        .value("WRITHING_MASS", ME::WRITHING_MASS)
        .value("GIANT_HEAD", ME::GIANT_HEAD)
        .value("NEMESIS", ME::NEMESIS)
        .value("REPTOMANCER", ME::REPTOMANCER)
        .value("AWAKENED_ONE", ME::AWAKENED_ONE)
        .value("TIME_EATER", ME::TIME_EATER)
        .value("DONU_AND_DECA", ME::DONU_AND_DECA)
        .value("SHIELD_AND_SPEAR", ME::SHIELD_AND_SPEAR)
        .value("THE_HEART", ME::THE_HEART)
        .value("LAGAVULIN_EVENT", ME::LAGAVULIN_EVENT)
        .value("COLOSSEUM_EVENT_SLAVERS", ME::COLOSSEUM_EVENT_SLAVERS)
        .value("COLOSSEUM_EVENT_NOBS", ME::COLOSSEUM_EVENT_NOBS)
        .value("MASKED_BANDITS_EVENT", ME::MASKED_BANDITS_EVENT)
        .value("MUSHROOMS_EVENT", ME::MUSHROOMS_EVENT)
        .value("MYSTERIOUS_SPHERE_EVENT", ME::MYSTERIOUS_SPHERE_EVENT);

    pybind11::enum_<RelicId> relicEnum(m, "RelicId");
    relicEnum.value("AKABEKO", RelicId::AKABEKO)
        .value("ART_OF_WAR", RelicId::ART_OF_WAR)
        .value("BIRD_FACED_URN", RelicId::BIRD_FACED_URN)
        .value("BLOODY_IDOL", RelicId::BLOODY_IDOL)
        .value("BLUE_CANDLE", RelicId::BLUE_CANDLE)
        .value("BRIMSTONE", RelicId::BRIMSTONE)
        .value("CALIPERS", RelicId::CALIPERS)
        .value("CAPTAINS_WHEEL", RelicId::CAPTAINS_WHEEL)
        .value("CENTENNIAL_PUZZLE", RelicId::CENTENNIAL_PUZZLE)
        .value("CERAMIC_FISH", RelicId::CERAMIC_FISH)
        .value("CHAMPION_BELT", RelicId::CHAMPION_BELT)
        .value("CHARONS_ASHES", RelicId::CHARONS_ASHES)
        .value("CHEMICAL_X", RelicId::CHEMICAL_X)
        .value("CLOAK_CLASP", RelicId::CLOAK_CLASP)
        .value("DARKSTONE_PERIAPT", RelicId::DARKSTONE_PERIAPT)
        .value("DEAD_BRANCH", RelicId::DEAD_BRANCH)
        .value("DUALITY", RelicId::DUALITY)
        .value("ECTOPLASM", RelicId::ECTOPLASM)
        .value("EMOTION_CHIP", RelicId::EMOTION_CHIP)
        .value("FROZEN_CORE", RelicId::FROZEN_CORE)
        .value("FROZEN_EYE", RelicId::FROZEN_EYE)
        .value("GAMBLING_CHIP", RelicId::GAMBLING_CHIP)
        .value("GINGER", RelicId::GINGER)
        .value("GOLDEN_EYE", RelicId::GOLDEN_EYE)
        .value("GREMLIN_HORN", RelicId::GREMLIN_HORN)
        .value("HAND_DRILL", RelicId::HAND_DRILL)
        .value("HAPPY_FLOWER", RelicId::HAPPY_FLOWER)
        .value("HORN_CLEAT", RelicId::HORN_CLEAT)
        .value("HOVERING_KITE", RelicId::HOVERING_KITE)
        .value("ICE_CREAM", RelicId::ICE_CREAM)
        .value("INCENSE_BURNER", RelicId::INCENSE_BURNER)
        .value("INK_BOTTLE", RelicId::INK_BOTTLE)
        .value("INSERTER", RelicId::INSERTER)
        .value("KUNAI", RelicId::KUNAI)
        .value("LETTER_OPENER", RelicId::LETTER_OPENER)
        .value("LIZARD_TAIL", RelicId::LIZARD_TAIL)
        .value("MAGIC_FLOWER", RelicId::MAGIC_FLOWER)
        .value("MARK_OF_THE_BLOOM", RelicId::MARK_OF_THE_BLOOM)
        .value("MEDICAL_KIT", RelicId::MEDICAL_KIT)
        .value("MELANGE", RelicId::MELANGE)
        .value("MERCURY_HOURGLASS", RelicId::MERCURY_HOURGLASS)
        .value("MUMMIFIED_HAND", RelicId::MUMMIFIED_HAND)
        .value("NECRONOMICON", RelicId::NECRONOMICON)
        .value("NILRYS_CODEX", RelicId::NILRYS_CODEX)
        .value("NUNCHAKU", RelicId::NUNCHAKU)
        .value("ODD_MUSHROOM", RelicId::ODD_MUSHROOM)
        .value("OMAMORI", RelicId::OMAMORI)
        .value("ORANGE_PELLETS", RelicId::ORANGE_PELLETS)
        .value("ORICHALCUM", RelicId::ORICHALCUM)
        .value("ORNAMENTAL_FAN", RelicId::ORNAMENTAL_FAN)
        .value("PAPER_KRANE", RelicId::PAPER_KRANE)
        .value("PAPER_PHROG", RelicId::PAPER_PHROG)
        .value("PEN_NIB", RelicId::PEN_NIB)
        .value("PHILOSOPHERS_STONE", RelicId::PHILOSOPHERS_STONE)
        .value("POCKETWATCH", RelicId::POCKETWATCH)
        .value("RED_SKULL", RelicId::RED_SKULL)
        .value("RUNIC_CUBE", RelicId::RUNIC_CUBE)
        .value("RUNIC_DOME", RelicId::RUNIC_DOME)
        .value("RUNIC_PYRAMID", RelicId::RUNIC_PYRAMID)
        .value("SACRED_BARK", RelicId::SACRED_BARK)
        .value("SELF_FORMING_CLAY", RelicId::SELF_FORMING_CLAY)
        .value("SHURIKEN", RelicId::SHURIKEN)
        .value("SNECKO_EYE", RelicId::SNECKO_EYE)
        .value("SNECKO_SKULL", RelicId::SNECKO_SKULL)
        .value("SOZU", RelicId::SOZU)
        .value("STONE_CALENDAR", RelicId::STONE_CALENDAR)
        .value("STRANGE_SPOON", RelicId::STRANGE_SPOON)
        .value("STRIKE_DUMMY", RelicId::STRIKE_DUMMY)
        .value("SUNDIAL", RelicId::SUNDIAL)
        .value("THE_ABACUS", RelicId::THE_ABACUS)
        .value("THE_BOOT", RelicId::THE_BOOT)
        .value("THE_SPECIMEN", RelicId::THE_SPECIMEN)
        .value("TINGSHA", RelicId::TINGSHA)
        .value("TOOLBOX", RelicId::TOOLBOX)
        .value("TORII", RelicId::TORII)
        .value("TOUGH_BANDAGES", RelicId::TOUGH_BANDAGES)
        .value("TOY_ORNITHOPTER", RelicId::TOY_ORNITHOPTER)
        .value("TUNGSTEN_ROD", RelicId::TUNGSTEN_ROD)
        .value("TURNIP", RelicId::TURNIP)
        .value("TWISTED_FUNNEL", RelicId::TWISTED_FUNNEL)
        .value("UNCEASING_TOP", RelicId::UNCEASING_TOP)
        .value("VELVET_CHOKER", RelicId::VELVET_CHOKER)
        .value("VIOLET_LOTUS", RelicId::VIOLET_LOTUS)
        .value("WARPED_TONGS", RelicId::WARPED_TONGS)
        .value("WRIST_BLADE", RelicId::WRIST_BLADE)
        .value("BLACK_BLOOD", RelicId::BLACK_BLOOD)
        .value("BURNING_BLOOD", RelicId::BURNING_BLOOD)
        .value("MEAT_ON_THE_BONE", RelicId::MEAT_ON_THE_BONE)
        .value("FACE_OF_CLERIC", RelicId::FACE_OF_CLERIC)
        .value("ANCHOR", RelicId::ANCHOR)
        .value("ANCIENT_TEA_SET", RelicId::ANCIENT_TEA_SET)
        .value("BAG_OF_MARBLES", RelicId::BAG_OF_MARBLES)
        .value("BAG_OF_PREPARATION", RelicId::BAG_OF_PREPARATION)
        .value("BLOOD_VIAL", RelicId::BLOOD_VIAL)
        .value("BOTTLED_FLAME", RelicId::BOTTLED_FLAME)
        .value("BOTTLED_LIGHTNING", RelicId::BOTTLED_LIGHTNING)
        .value("BOTTLED_TORNADO", RelicId::BOTTLED_TORNADO)
        .value("BRONZE_SCALES", RelicId::BRONZE_SCALES)
        .value("BUSTED_CROWN", RelicId::BUSTED_CROWN)
        .value("CLOCKWORK_SOUVENIR", RelicId::CLOCKWORK_SOUVENIR)
        .value("COFFEE_DRIPPER", RelicId::COFFEE_DRIPPER)
        .value("CRACKED_CORE", RelicId::CRACKED_CORE)
        .value("CURSED_KEY", RelicId::CURSED_KEY)
        .value("DAMARU", RelicId::DAMARU)
        .value("DATA_DISK", RelicId::DATA_DISK)
        .value("DU_VU_DOLL", RelicId::DU_VU_DOLL)
        .value("ENCHIRIDION", RelicId::ENCHIRIDION)
        .value("FOSSILIZED_HELIX", RelicId::FOSSILIZED_HELIX)
        .value("FUSION_HAMMER", RelicId::FUSION_HAMMER)
        .value("GIRYA", RelicId::GIRYA)
        .value("GOLD_PLATED_CABLES", RelicId::GOLD_PLATED_CABLES)
        .value("GREMLIN_VISAGE", RelicId::GREMLIN_VISAGE)
        .value("HOLY_WATER", RelicId::HOLY_WATER)
        .value("LANTERN", RelicId::LANTERN)
        .value("MARK_OF_PAIN", RelicId::MARK_OF_PAIN)
        .value("MUTAGENIC_STRENGTH", RelicId::MUTAGENIC_STRENGTH)
        .value("NEOWS_LAMENT", RelicId::NEOWS_LAMENT)
        .value("NINJA_SCROLL", RelicId::NINJA_SCROLL)
        .value("NUCLEAR_BATTERY", RelicId::NUCLEAR_BATTERY)
        .value("ODDLY_SMOOTH_STONE", RelicId::ODDLY_SMOOTH_STONE)
        .value("PANTOGRAPH", RelicId::PANTOGRAPH)
        .value("PRESERVED_INSECT", RelicId::PRESERVED_INSECT)
        .value("PURE_WATER", RelicId::PURE_WATER)
        .value("RED_MASK", RelicId::RED_MASK)
        .value("RING_OF_THE_SERPENT", RelicId::RING_OF_THE_SERPENT)
        .value("RING_OF_THE_SNAKE", RelicId::RING_OF_THE_SNAKE)
        .value("RUNIC_CAPACITOR", RelicId::RUNIC_CAPACITOR)
        .value("SLAVERS_COLLAR", RelicId::SLAVERS_COLLAR)
        .value("SLING_OF_COURAGE", RelicId::SLING_OF_COURAGE)
        .value("SYMBIOTIC_VIRUS", RelicId::SYMBIOTIC_VIRUS)
        .value("TEARDROP_LOCKET", RelicId::TEARDROP_LOCKET)
        .value("THREAD_AND_NEEDLE", RelicId::THREAD_AND_NEEDLE)
        .value("VAJRA", RelicId::VAJRA)
        .value("ASTROLABE", RelicId::ASTROLABE)
        .value("BLACK_STAR", RelicId::BLACK_STAR)
        .value("CALLING_BELL", RelicId::CALLING_BELL)
        .value("CAULDRON", RelicId::CAULDRON)
        .value("CULTIST_HEADPIECE", RelicId::CULTIST_HEADPIECE)
        .value("DOLLYS_MIRROR", RelicId::DOLLYS_MIRROR)
        .value("DREAM_CATCHER", RelicId::DREAM_CATCHER)
        .value("EMPTY_CAGE", RelicId::EMPTY_CAGE)
        .value("ETERNAL_FEATHER", RelicId::ETERNAL_FEATHER)
        .value("FROZEN_EGG", RelicId::FROZEN_EGG)
        .value("GOLDEN_IDOL", RelicId::GOLDEN_IDOL)
        .value("JUZU_BRACELET", RelicId::JUZU_BRACELET)
        .value("LEES_WAFFLE", RelicId::LEES_WAFFLE)
        .value("MANGO", RelicId::MANGO)
        .value("MATRYOSHKA", RelicId::MATRYOSHKA)
        .value("MAW_BANK", RelicId::MAW_BANK)
        .value("MEAL_TICKET", RelicId::MEAL_TICKET)
        .value("MEMBERSHIP_CARD", RelicId::MEMBERSHIP_CARD)
        .value("MOLTEN_EGG", RelicId::MOLTEN_EGG)
        .value("NLOTHS_GIFT", RelicId::NLOTHS_GIFT)
        .value("NLOTHS_HUNGRY_FACE", RelicId::NLOTHS_HUNGRY_FACE)
        .value("OLD_COIN", RelicId::OLD_COIN)
        .value("ORRERY", RelicId::ORRERY)
        .value("PANDORAS_BOX", RelicId::PANDORAS_BOX)
        .value("PEACE_PIPE", RelicId::PEACE_PIPE)
        .value("PEAR", RelicId::PEAR)
        .value("POTION_BELT", RelicId::POTION_BELT)
        .value("PRAYER_WHEEL", RelicId::PRAYER_WHEEL)
        .value("PRISMATIC_SHARD", RelicId::PRISMATIC_SHARD)
        .value("QUESTION_CARD", RelicId::QUESTION_CARD)
        .value("REGAL_PILLOW", RelicId::REGAL_PILLOW)
        .value("SSSERPENT_HEAD", RelicId::SSSERPENT_HEAD)
        .value("SHOVEL", RelicId::SHOVEL)
        .value("SINGING_BOWL", RelicId::SINGING_BOWL)
        .value("SMILING_MASK", RelicId::SMILING_MASK)
        .value("SPIRIT_POOP", RelicId::SPIRIT_POOP)
        .value("STRAWBERRY", RelicId::STRAWBERRY)
        .value("THE_COURIER", RelicId::THE_COURIER)
        .value("TINY_CHEST", RelicId::TINY_CHEST)
        .value("TINY_HOUSE", RelicId::TINY_HOUSE)
        .value("TOXIC_EGG", RelicId::TOXIC_EGG)
        .value("WAR_PAINT", RelicId::WAR_PAINT)
        .value("WHETSTONE", RelicId::WHETSTONE)
        .value("WHITE_BEAST_STATUE", RelicId::WHITE_BEAST_STATUE)
        .value("WING_BOOTS", RelicId::WING_BOOTS)
        .value("CIRCLET", RelicId::CIRCLET)
        .value("RED_CIRCLET", RelicId::RED_CIRCLET)
        .value("INVALID", RelicId::INVALID);

    pybind11::enum_<Potion>(m, "Potion")
        .value("INVALID", Potion::INVALID)
        .value("EMPTY_POTION_SLOT", Potion::EMPTY_POTION_SLOT)
        .value("AMBROSIA", Potion::AMBROSIA)
        .value("ANCIENT_POTION", Potion::ANCIENT_POTION)
        .value("ATTACK_POTION", Potion::ATTACK_POTION)
        .value("BLESSING_OF_THE_FORGE", Potion::BLESSING_OF_THE_FORGE)
        .value("BLOCK_POTION", Potion::BLOCK_POTION)
        .value("BLOOD_POTION", Potion::BLOOD_POTION)
        .value("BOTTLED_MIRACLE", Potion::BOTTLED_MIRACLE)
        .value("COLORLESS_POTION", Potion::COLORLESS_POTION)
        .value("CULTIST_POTION", Potion::CULTIST_POTION)
        .value("CUNNING_POTION", Potion::CUNNING_POTION)
        .value("DEXTERITY_POTION", Potion::DEXTERITY_POTION)
        .value("DISTILLED_CHAOS", Potion::DISTILLED_CHAOS)
        .value("DUPLICATION_POTION", Potion::DUPLICATION_POTION)
        .value("ELIXIR_POTION", Potion::ELIXIR_POTION)
        .value("ENERGY_POTION", Potion::ENERGY_POTION)
        .value("ENTROPIC_BREW", Potion::ENTROPIC_BREW)
        .value("ESSENCE_OF_DARKNESS", Potion::ESSENCE_OF_DARKNESS)
        .value("ESSENCE_OF_STEEL", Potion::ESSENCE_OF_STEEL)
        .value("EXPLOSIVE_POTION", Potion::EXPLOSIVE_POTION)
        .value("FAIRY_POTION", Potion::FAIRY_POTION)
        .value("FEAR_POTION", Potion::FEAR_POTION)
        .value("FIRE_POTION", Potion::FIRE_POTION)
        .value("FLEX_POTION", Potion::FLEX_POTION)
        .value("FOCUS_POTION", Potion::FOCUS_POTION)
        .value("FRUIT_JUICE", Potion::FRUIT_JUICE)
        .value("GAMBLERS_BREW", Potion::GAMBLERS_BREW)
        .value("GHOST_IN_A_JAR", Potion::GHOST_IN_A_JAR)
        .value("HEART_OF_IRON", Potion::HEART_OF_IRON)
        .value("LIQUID_BRONZE", Potion::LIQUID_BRONZE)
        .value("LIQUID_MEMORIES", Potion::LIQUID_MEMORIES)
        .value("POISON_POTION", Potion::POISON_POTION)
        .value("POTION_OF_CAPACITY", Potion::POTION_OF_CAPACITY)
        .value("POWER_POTION", Potion::POWER_POTION)
        .value("REGEN_POTION", Potion::REGEN_POTION)
        .value("SKILL_POTION", Potion::SKILL_POTION)
        .value("SMOKE_BOMB", Potion::SMOKE_BOMB)
        .value("SNECKO_OIL", Potion::SNECKO_OIL)
        .value("SPEED_POTION", Potion::SPEED_POTION)
        .value("STANCE_POTION", Potion::STANCE_POTION)
        .value("STRENGTH_POTION", Potion::STRENGTH_POTION)
        .value("SWIFT_POTION", Potion::SWIFT_POTION)
        .value("WEAK_POTION", Potion::WEAK_POTION);

    pybind11::enum_<CardSelectScreenType>(m, "CardSelectScreenType")
        .value("INVALID", CardSelectScreenType::INVALID)
        .value("TRANSFORM", CardSelectScreenType::TRANSFORM)
        .value("TRANSFORM_UPGRADE", CardSelectScreenType::TRANSFORM_UPGRADE)
        .value("UPGRADE", CardSelectScreenType::UPGRADE)
        .value("REMOVE", CardSelectScreenType::REMOVE)
        .value("DUPLICATE", CardSelectScreenType::DUPLICATE)
        .value("OBTAIN", CardSelectScreenType::OBTAIN)
        .value("BOTTLE", CardSelectScreenType::BOTTLE)
        .value("BONFIRE_SPIRITS", CardSelectScreenType::BONFIRE_SPIRITS);

    // Neow enums exposed as flat names on the module
    pybind11::enum_<Neow::Bonus>(m, "NeowBonus")
        .value("THREE_CARDS", Neow::Bonus::THREE_CARDS)
        .value("ONE_RANDOM_RARE_CARD", Neow::Bonus::ONE_RANDOM_RARE_CARD)
        .value("REMOVE_CARD", Neow::Bonus::REMOVE_CARD)
        .value("UPGRADE_CARD", Neow::Bonus::UPGRADE_CARD)
        .value("TRANSFORM_CARD", Neow::Bonus::TRANSFORM_CARD)
        .value("RANDOM_COLORLESS", Neow::Bonus::RANDOM_COLORLESS)
        .value("THREE_SMALL_POTIONS", Neow::Bonus::THREE_SMALL_POTIONS)
        .value("RANDOM_COMMON_RELIC", Neow::Bonus::RANDOM_COMMON_RELIC)
        .value("TEN_PERCENT_HP_BONUS", Neow::Bonus::TEN_PERCENT_HP_BONUS)
        .value("THREE_ENEMY_KILL", Neow::Bonus::THREE_ENEMY_KILL)
        .value("HUNDRED_GOLD", Neow::Bonus::HUNDRED_GOLD)
        .value("RANDOM_COLORLESS_2", Neow::Bonus::RANDOM_COLORLESS_2)
        .value("REMOVE_TWO", Neow::Bonus::REMOVE_TWO)
        .value("ONE_RARE_RELIC", Neow::Bonus::ONE_RARE_RELIC)
        .value("THREE_RARE_CARDS", Neow::Bonus::THREE_RARE_CARDS)
        .value("TWO_FIFTY_GOLD", Neow::Bonus::TWO_FIFTY_GOLD)
        .value("TRANSFORM_TWO_CARDS", Neow::Bonus::TRANSFORM_TWO_CARDS)
        .value("TWENTY_PERCENT_HP_BONUS", Neow::Bonus::TWENTY_PERCENT_HP_BONUS)
        .value("BOSS_RELIC", Neow::Bonus::BOSS_RELIC)
        .value("INVALID", Neow::Bonus::INVALID);

    pybind11::enum_<Neow::Drawback>(m, "NeowCost")
        .value("INVALID", Neow::Drawback::INVALID)
        .value("NONE", Neow::Drawback::NONE)
        .value("TEN_PERCENT_HP_LOSS", Neow::Drawback::TEN_PERCENT_HP_LOSS)
        .value("NO_GOLD", Neow::Drawback::NO_GOLD)
        .value("CURSE", Neow::Drawback::CURSE)
        .value("PERCENT_DAMAGE", Neow::Drawback::PERCENT_DAMAGE)
        .value("LOSE_STARTER_RELIC", Neow::Drawback::LOSE_STARTER_RELIC);

    // -----------------------------------------------------------------------
    // InputState enum
    // -----------------------------------------------------------------------
    pybind11::enum_<InputState>(m, "InputState")
        .value("EXECUTING_ACTIONS", InputState::EXECUTING_ACTIONS)
        .value("PLAYER_NORMAL", InputState::PLAYER_NORMAL)
        .value("CARD_SELECT", InputState::CARD_SELECT)
        .value("CHOOSE_STANCE_ACTION", InputState::CHOOSE_STANCE_ACTION)
        .value("CHOOSE_TOOLBOX_COLORLESS_CARD", InputState::CHOOSE_TOOLBOX_COLORLESS_CARD)
        .value("CHOOSE_EXHAUST_POTION_CARDS", InputState::CHOOSE_EXHAUST_POTION_CARDS)
        .value("CHOOSE_GAMBLING_CARDS", InputState::CHOOSE_GAMBLING_CARDS)
        .value("CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS", InputState::CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS)
        .value("CHOOSE_DISCARD_CARDS", InputState::CHOOSE_DISCARD_CARDS)
        .value("SCRY", InputState::SCRY)
        .value("SELECT_ENEMY_ACTIONS", InputState::SELECT_ENEMY_ACTIONS)
        .value("FILL_RANDOM_POTIONS", InputState::FILL_RANDOM_POTIONS)
        .value("SHUFFLE_INTO_DRAW_BURN", InputState::SHUFFLE_INTO_DRAW_BURN)
        .value("SHUFFLE_INTO_DRAW_VOID", InputState::SHUFFLE_INTO_DRAW_VOID)
        .value("SHUFFLE_INTO_DRAW_DAZED", InputState::SHUFFLE_INTO_DRAW_DAZED)
        .value("SHUFFLE_INTO_DRAW_WOUND", InputState::SHUFFLE_INTO_DRAW_WOUND)
        .value("SHUFFLE_INTO_DRAW_SLIMED", InputState::SHUFFLE_INTO_DRAW_SLIMED)
        .value("SHUFFLE_INTO_DRAW_ALL_STATUS", InputState::SHUFFLE_INTO_DRAW_ALL_STATUS)
        .value("SHUFFLE_CUR_CARD_INTO_DRAW", InputState::SHUFFLE_CUR_CARD_INTO_DRAW)
        .value("SHUFFLE_DISCARD_TO_DRAW", InputState::SHUFFLE_DISCARD_TO_DRAW)
        .value("INITIAL_SHUFFLE", InputState::INITIAL_SHUFFLE)
        .value("CREATE_RANDOM_CARD_IN_HAND_POWER", InputState::CREATE_RANDOM_CARD_IN_HAND_POWER)
        .value("CREATE_RANDOM_CARD_IN_HAND_COLORLESS", InputState::CREATE_RANDOM_CARD_IN_HAND_COLORLESS)
        .value("CREATE_RANDOM_CARD_IN_HAND_DEAD_BRANCH", InputState::CREATE_RANDOM_CARD_IN_HAND_DEAD_BRANCH)
        .value("SELECT_CARD_IN_HAND_EXHAUST", InputState::SELECT_CARD_IN_HAND_EXHAUST)
        .value("GENERATE_NILRY_CARDS", InputState::GENERATE_NILRY_CARDS)
        .value("EXHAUST_RANDOM_CARD_IN_HAND", InputState::EXHAUST_RANDOM_CARD_IN_HAND)
        .value("SELECT_STRANGE_SPOON_PROC", InputState::SELECT_STRANGE_SPOON_PROC)
        .value("SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON", InputState::SELECT_ENEMY_THE_SPECIMEN_APPLY_POISON)
        .value("SELECT_WARPED_TONGS_CARD", InputState::SELECT_WARPED_TONGS_CARD)
        .value("CREATE_ENCHIRIDION_POWER", InputState::CREATE_ENCHIRIDION_POWER)
        .value("SELECT_CONFUSED_CARD_COST", InputState::SELECT_CONFUSED_CARD_COST);

    // -----------------------------------------------------------------------
    // CardSelectTask enum
    // -----------------------------------------------------------------------
    pybind11::enum_<CardSelectTask>(m, "CardSelectTask")
        .value("INVALID", CardSelectTask::INVALID)
        .value("ARMAMENTS", CardSelectTask::ARMAMENTS)
        .value("CODEX", CardSelectTask::CODEX)
        .value("DISCOVERY", CardSelectTask::DISCOVERY)
        .value("DUAL_WIELD", CardSelectTask::DUAL_WIELD)
        .value("EXHAUST_ONE", CardSelectTask::EXHAUST_ONE)
        .value("EXHAUST_MANY", CardSelectTask::EXHAUST_MANY)
        .value("EXHUME", CardSelectTask::EXHUME)
        .value("FORETHOUGHT", CardSelectTask::FORETHOUGHT)
        .value("GAMBLE", CardSelectTask::GAMBLE)
        .value("HEADBUTT", CardSelectTask::HEADBUTT)
        .value("HOLOGRAM", CardSelectTask::HOLOGRAM)
        .value("LIQUID_MEMORIES_POTION", CardSelectTask::LIQUID_MEMORIES_POTION)
        .value("MEDITATE", CardSelectTask::MEDITATE)
        .value("NIGHTMARE", CardSelectTask::NIGHTMARE)
        .value("RECYCLE", CardSelectTask::RECYCLE)
        .value("SECRET_TECHNIQUE", CardSelectTask::SECRET_TECHNIQUE)
        .value("SECRET_WEAPON", CardSelectTask::SECRET_WEAPON)
        .value("SEEK", CardSelectTask::SEEK)
        .value("SETUP", CardSelectTask::SETUP)
        .value("WARCRY", CardSelectTask::WARCRY);

    // -----------------------------------------------------------------------
    // BattleContext — exposed for Python micro-agent step-by-step control.
    //
    // Usage pattern:
    //   bc = sts.BattleContext()
    //   bc.init(gc)
    //   bc.execute_actions()            # advance past initial shuffle / innates
    //   while not bc.is_over:
    //       state_dict = bc.get_state() # read state
    //       if bc.input_state == sts.InputState.PLAYER_NORMAL:
    //           bc.play_card(hand_idx, target_idx)  # or bc.end_turn()
    //       elif bc.input_state == sts.InputState.CARD_SELECT:
    //           bc.choose_hand_card(hand_idx)        # or choose_discard_card etc.
    //       bc.execute_actions()
    //   bc.exit_battle(gc)
    // -----------------------------------------------------------------------

    // Helper lambda: convert a CardInstance to a Python dict
    auto card_to_dict = [](const CardInstance &c) -> pybind11::dict {
        pybind11::dict d;
        d["card_id"]      = std::string(cardStringIds[static_cast<int>(c.id)]);
        d["upgraded"]     = c.isUpgraded();
        d["cost"]         = static_cast<int>(c.cost);
        d["cost_for_turn"]= static_cast<int>(c.costForTurn);
        d["special_data"] = static_cast<int>(c.specialData);
        d["unique_id"]    = static_cast<int>(c.uniqueId);
        return d;
    };

    // Helper lambda: convert a Monster to a Python dict
    // bc is needed to compute calculateDamageToPlayer for intent_damage.
    auto monster_to_dict = [](const Monster &m, const BattleContext &bc) -> pybind11::dict {
        pybind11::dict d;
        d["monster_id"]       = std::string(monsterIdStrings[static_cast<int>(m.id)]);
        d["current_hp"]       = m.curHp;
        d["max_hp"]           = m.maxHp;
        d["block"]            = m.block;
        d["move_id"]          = std::string(monsterMoveStrings[static_cast<int>(m.moveHistory[0])]);
        d["move_history_0"]   = std::string(monsterMoveStrings[static_cast<int>(m.moveHistory[0])]);
        d["move_history_1"]   = std::string(monsterMoveStrings[static_cast<int>(m.moveHistory[1])]);
        // Note: current intended move is whatever was last set via setMove(); we expose move_history[0]
        // as the "current" move since the game writes the current move there before executing it.
        // Intent damage: displayed per-hit damage after all modifiers (monster strength, weak, player vulnerable, etc.)
        {
            DamageInfo di = m.getMoveBaseDamage(bc);
            if (di.attackCount > 0 && di.damage > 0) {
                d["intent_damage"]    = m.calculateDamageToPlayer(bc, di.damage);
                d["intent_hit_count"] = di.attackCount;
            } else {
                d["intent_damage"]    = 0;
                d["intent_hit_count"] = 0;
            }
        }
        // Status effects
        d["strength"]         = m.strength;
        d["vulnerable"]       = m.vulnerable;
        d["weak"]             = m.weak;
        d["artifact"]         = static_cast<int>(m.artifact);
        d["poison"]           = static_cast<int>(m.poison);
        d["metallicize"]      = static_cast<int>(m.metallicize);
        d["plated_armor"]     = static_cast<int>(m.platedArmor);
        d["regen"]            = static_cast<int>(m.regen);
        d["block_return"]     = static_cast<int>(m.blockReturn);
        d["choked"]           = static_cast<int>(m.choked);
        d["corpse_explosion"] = static_cast<int>(m.corpseExplosion);
        d["lock_on"]          = static_cast<int>(m.lockOn);
        d["mark"]             = static_cast<int>(m.mark);
        d["shackled"]         = static_cast<int>(m.shackled);
        d["unique_power0"]    = m.uniquePower0;
        d["unique_power1"]    = static_cast<int>(m.uniquePower1);
        // Boolean statuses (packed in statusBits)
        d["asleep"]       = m.hasStatus<MS::ASLEEP>();
        d["barricade"]    = m.hasStatus<MS::BARRICADE>();
        d["minion"]       = m.hasStatus<MS::MINION>();
        d["minion_leader"]= m.hasStatus<MS::MINION_LEADER>();
        d["painful_stabs"]= m.hasStatus<MS::PAINFUL_STABS>();
        d["regrow"]       = m.hasStatus<MS::REGROW>();
        d["shifting"]     = m.hasStatus<MS::SHIFTING>();
        d["stasis"]       = m.hasStatus<MS::STASIS>();
        // Flags
        d["is_alive"]     = m.isAlive();
        d["half_dead"]    = m.halfDead;
        d["is_escaping"]  = m.isEscaping();
        return d;
    };

    pybind11::class_<BattleContext>(m, "BattleContext")
        .def(pybind11::init<>())

        // Initialise from a GameContext (starts the fight)
        .def("init",
            [](BattleContext &bc, GameContext &gc) { bc.init(gc); },
            "Initialise BattleContext from a GameContext (sets up monsters, shuffles deck, etc.)")

        // Advance until a player-decision point is reached (PLAYER_NORMAL or CARD_SELECT)
        // or the battle ends (isBattleOver == true).
        .def("execute_actions",
            [](BattleContext &bc) { bc.executeActions(); },
            "Run the action queue until a player decision is needed or the battle ends")

        // Primary decision: play a card from hand
        .def("play_card",
            [](BattleContext &bc, int hand_idx, int target_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                const CardInstance &card = bc.cards.hand[hand_idx];
                bc.addToBotCard(CardQueueItem(card, target_idx, bc.player.energy));
                bc.inputState = InputState::EXECUTING_ACTIONS;
                bc.executeActions();
            },
            pybind11::arg("hand_idx"),
            pybind11::arg("target_idx") = 0,
            "Play the card at hand_idx targeting monster at target_idx (default 0)")

        // End the current player turn
        .def("end_turn",
            [](BattleContext &bc) {
                bc.endTurn();
                bc.inputState = InputState::EXECUTING_ACTIONS;
                bc.executeActions();
            },
            "End the current player turn (also pumps the action queue)")

        // Resume action queue processing: resets inputState to EXECUTING_ACTIONS
        // and calls executeActions(). Use this after 'automatic' intermediate states
        // (e.g. INITIAL_SHUFFLE, FILL_RANDOM_POTIONS) that are not PLAYER_NORMAL or
        // CARD_SELECT and do not require Python input.
        .def("resume_actions",
            [](BattleContext &bc) {
                bc.inputState = InputState::EXECUTING_ACTIONS;
                bc.executeActions();
            },
            "Reset inputState to EXECUTING_ACTIONS and pump the action queue")

        // Card-select screen choices (CARD_SELECT input state)
        .def("choose_armaments_card",
            [](BattleContext &bc, int hand_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                bc.chooseArmamentsCard(hand_idx);
            },
            "Choose a card in hand to upgrade (Armaments)")
        .def("choose_discovery_card",
            [](BattleContext &bc, int card_id_int) {
                bc.chooseDiscoveryCard(static_cast<CardId>(card_id_int));
            },
            "Choose one of three Discovery card options by CardId int")
        .def("choose_dual_wield_card",
            [](BattleContext &bc, int hand_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                bc.chooseDualWieldCard(hand_idx);
            },
            "Choose a card in hand to duplicate (Dual Wield)")
        .def("choose_exhaust_one_card",
            [](BattleContext &bc, int hand_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                bc.chooseExhaustOneCard(hand_idx);
            },
            "Exhaust a card in hand (Exhaust One select screen)")
        .def("choose_exhume_card",
            [](BattleContext &bc, int exhaust_idx) { bc.chooseExhumeCard(exhaust_idx); },
            "Exhume a card from the exhaust pile")
        .def("choose_forethought_card",
            [](BattleContext &bc, int hand_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                bc.chooseForethoughtCard(hand_idx);
            },
            "Choose a card in hand to place on bottom of draw pile (Forethought)")
        .def("choose_headbutt_card",
            [](BattleContext &bc, int discard_idx) { bc.chooseHeadbuttCard(discard_idx); },
            "Choose a card in discard pile to put on top of draw pile (Headbutt)")
        .def("choose_recycle_card",
            [](BattleContext &bc, int hand_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                bc.chooseRecycleCard(hand_idx);
            },
            "Choose a card in hand to recycle (gain energy equal to cost)")
        .def("choose_warcry_card",
            [](BattleContext &bc, int hand_idx) {
                if (hand_idx < 0 || hand_idx >= bc.cards.cardsInHand) return;
                bc.chooseWarcryCard(hand_idx);
            },
            "Choose a card in hand to put on top of draw pile (Warcry)")
        .def("choose_discard_to_hand_card",
            [](BattleContext &bc, int discard_idx, bool for_zero_cost) {
                bc.chooseDiscardToHandCard(discard_idx, for_zero_cost);
            },
            pybind11::arg("discard_idx"),
            pybind11::arg("for_zero_cost") = false,
            "Put a card from discard pile into hand")
        .def("choose_exhaust_many_cards",
            [](BattleContext &bc, std::vector<int> idxs) {
                fixed_list<int, 10> fl;
                for (int i : idxs) fl.push_back(i);
                bc.chooseExhaustCards(fl);
            },
            "Exhaust multiple cards from hand (Exhaust Many select screen)")
        .def("choose_gamble_cards",
            [](BattleContext &bc, std::vector<int> idxs) {
                fixed_list<int, 10> fl;
                for (int i : idxs) fl.push_back(i);
                bc.chooseGambleCards(fl);
            },
            "Keep the specified hand indices; discard the rest (Gamble)")

        // Use a potion from a potion slot
        .def("drink_potion",
            [](BattleContext &bc, int slot_idx, int target_idx) {
                if (slot_idx < 0 || slot_idx >= bc.potionCapacity) return;
                if (bc.potions[slot_idx] == Potion::EMPTY_POTION_SLOT) return;
                if (bc.potions[slot_idx] == Potion::FAIRY_POTION) return;  // auto-triggers on death; cannot be manually used
                bc.drinkPotion(slot_idx, target_idx);
                bc.inputState = InputState::EXECUTING_ACTIONS;
                bc.executeActions();
            },
            pybind11::arg("slot_idx"),
            pybind11::arg("target_idx") = 0,
            "Use the potion in slot_idx targeting monster at target_idx (default 0); pumps the action queue")

        // Exit battle and propagate results back to GameContext
        .def("exit_battle",
            [](BattleContext &bc, GameContext &gc) { bc.exitBattle(gc); },
            "Propagate combat results back to the GameContext after battle ends")

        // State query properties
        .def_property_readonly("input_state",
            [](const BattleContext &bc) { return bc.inputState; },
            "Current InputState (PLAYER_NORMAL, CARD_SELECT, etc.)")
        .def_property_readonly("outcome",
            [](const BattleContext &bc) { return static_cast<int>(bc.outcome); },
            "Current Outcome as int (0=UNDECIDED, 1=PLAYER_VICTORY, 2=PLAYER_LOSS)")
        .def_property_readonly("is_over",
            [](const BattleContext &bc) { return bc.outcome != Outcome::UNDECIDED; },
            "True if the battle has ended (victory or defeat)")
        .def_property_readonly("turn",
            [](const BattleContext &bc) { return bc.turn; },
            "Current turn number")
        .def_property_readonly("encounter",
            [](const BattleContext &bc) { return bc.encounter; },
            "MonsterEncounter for this fight")
        .def_property_readonly("ascension",
            [](const BattleContext &bc) { return bc.ascension; },
            "Ascension level")

        // get_card_select_info — analogous to GameContext.get_card_select_info()
        .def("get_card_select_info",
            [](const BattleContext &bc) -> pybind11::dict {
                pybind11::dict d;
                const auto &csi = bc.cardSelectInfo;
                d["task"]             = static_cast<int>(csi.cardSelectTask);
                d["pick_count"]       = csi.pickCount;
                d["can_pick_zero"]    = csi.canPickZero;
                d["can_pick_any"]     = csi.canPickAnyNumber;
                // cards field: only meaningful for DISCOVERY and CODEX
                pybind11::list cards;
                for (int i = 0; i < 3; ++i) {
                    cards.append(static_cast<int>(csi.cards[i]));
                }
                d["cards"] = cards;
                return d;
            },
            "Return card-select screen info (task, pick_count, can_pick_zero, cards)")

        // get_state — bulk extraction of complete battle state as a Python dict.
        // Called by LightspeedCombatAdapter.extract().
        .def("get_state",
            [card_to_dict, monster_to_dict](const BattleContext &bc) -> pybind11::dict {
                pybind11::dict d;
                const Player &p = bc.player;

                // --- Player core ---
                d["current_hp"]          = p.curHp;
                d["max_hp"]              = p.maxHp;
                d["block"]               = p.block;
                d["energy"]              = p.energy;
                d["energy_per_turn"]     = static_cast<int>(p.energyPerTurn);
                d["card_draw_per_turn"]  = static_cast<int>(p.cardDrawPerTurn);
                d["gold"]                = static_cast<int>(p.gold);
                d["character"]           = std::string(characterClassEnumNames[static_cast<int>(p.cc)]);
                d["ascension"]           = bc.ascension;
                d["floor_num"]           = bc.floorNum;
                d["turn"]                = bc.turn;

                // --- Player scalar statuses ---
                d["strength"]   = p.strength;
                d["dexterity"]  = p.dexterity;
                d["focus"]      = p.focus;
                d["artifact"]   = p.artifact;

                // --- Player stance / orbs ---
                d["stance"]    = std::string(stanceStrings[static_cast<int>(p.stance)]);
                d["orb_slots"] = static_cast<int>(p.orbSlots);

                // --- Player relic counters ---
                d["happy_flower_counter"]   = static_cast<int>(p.happyFlowerCounter);
                d["incense_burner_counter"] = static_cast<int>(p.incenseBurnerCounter);
                d["ink_bottle_counter"]     = static_cast<int>(p.inkBottleCounter);
                d["inserter_counter"]       = static_cast<int>(p.inserterCounter);
                d["nunchaku_counter"]       = static_cast<int>(p.nunchakuCounter);
                d["pen_nib_counter"]        = static_cast<int>(p.penNibCounter);
                d["sundial_counter"]        = static_cast<int>(p.sundialCounter);

                // --- Player internal counters ---
                d["bomb1"]                           = static_cast<int>(p.bomb1);
                d["bomb2"]                           = static_cast<int>(p.bomb2);
                d["bomb3"]                           = static_cast<int>(p.bomb3);
                d["combust_hp_loss"]                 = static_cast<int>(p.combustHpLoss);
                d["deva_form_energy_per_turn"]        = static_cast<int>(p.devaFormEnergyPerTurn);
                d["echo_form_cards_doubled"]          = static_cast<int>(p.echoFormCardsDoubled);
                d["panache_counter"]                  = static_cast<int>(p.panacheCounter);
                d["have_used_necronomicon_this_turn"] = p.haveUsedNecronomiconThisTurn;

                // --- Turn tracking ---
                d["cards_played_this_turn"]    = static_cast<int>(p.cardsPlayedThisTurn);
                d["attacks_played_this_turn"]  = static_cast<int>(p.attacksPlayedThisTurn);
                d["skills_played_this_turn"]   = static_cast<int>(p.skillsPlayedThisTurn);
                d["cards_discarded_this_turn"] = static_cast<int>(p.cardsDiscardedThisTurn);
                d["orange_pellets_attack"]     = (bool)p.orangePelletsCardTypesPlayed[0];
                d["orange_pellets_skill"]      = (bool)p.orangePelletsCardTypesPlayed[1];
                d["orange_pellets_power"]      = (bool)p.orangePelletsCardTypesPlayed[2];

                // --- Player status effects (all, accessed via hasStatus/getStatus) ---
                // We build a dict of {status_name: value} for all non-trivial statuses.
                // The adapter maps these to CombatState fields.
                pybind11::dict statuses;
                statuses["DOUBLE_DAMAGE"]       = (bool)p.hasStatus<PS::DOUBLE_DAMAGE>();
                statuses["DRAW_REDUCTION"]      = (bool)p.hasStatus<PS::DRAW_REDUCTION>();
                statuses["FRAIL"]               = p.getStatus<PS::FRAIL>();
                statuses["INTANGIBLE"]          = p.getStatus<PS::INTANGIBLE>();
                statuses["VULNERABLE"]          = p.getStatus<PS::VULNERABLE>();
                statuses["WEAK"]                = p.getStatus<PS::WEAK>();
                statuses["BIAS"]                = p.getStatus<PS::BIAS>();
                statuses["CONFUSED"]            = (bool)p.hasStatus<PS::CONFUSED>();
                statuses["CONSTRICTED"]         = p.getStatus<PS::CONSTRICTED>();
                statuses["ENTANGLED"]           = (bool)p.hasStatus<PS::ENTANGLED>();
                statuses["FASTING"]             = p.getStatus<PS::FASTING>();
                statuses["HEX"]                 = (bool)p.hasStatus<PS::HEX>();
                statuses["LOSE_DEXTERITY"]      = p.getStatus<PS::LOSE_DEXTERITY>();
                statuses["LOSE_STRENGTH"]       = p.getStatus<PS::LOSE_STRENGTH>();
                statuses["NO_BLOCK"]            = (bool)p.hasStatus<PS::NO_BLOCK>();
                statuses["NO_DRAW"]             = (bool)p.hasStatus<PS::NO_DRAW>();
                statuses["WRAITH_FORM"]         = p.getStatus<PS::WRAITH_FORM>();
                statuses["BARRICADE"]           = (bool)p.hasStatus<PS::BARRICADE>();
                statuses["BLASPHEMER"]          = (bool)p.hasStatus<PS::BLASPHEMER>();
                statuses["CORRUPTION"]          = (bool)p.hasStatus<PS::CORRUPTION>();
                statuses["ELECTRO"]             = (bool)p.hasStatus<PS::ELECTRO>();
                statuses["SURROUNDED"]          = (bool)p.hasStatus<PS::SURROUNDED>();
                statuses["MASTER_REALITY"]      = (bool)p.hasStatus<PS::MASTER_REALITY>();
                statuses["PEN_NIB"]             = (bool)p.hasStatus<PS::PEN_NIB>();
                statuses["WRATH_NEXT_TURN"]     = (bool)p.hasStatus<PS::WRATH_NEXT_TURN>();
                statuses["AMPLIFY"]             = p.getStatus<PS::AMPLIFY>();
                statuses["BLUR"]                = p.getStatus<PS::BLUR>();
                statuses["BUFFER"]              = p.getStatus<PS::BUFFER>();
                statuses["COLLECT"]             = p.getStatus<PS::COLLECT>();
                statuses["DOUBLE_TAP"]          = p.getStatus<PS::DOUBLE_TAP>();
                statuses["DUPLICATION"]         = p.getStatus<PS::DUPLICATION>();
                statuses["ECHO_FORM"]           = p.getStatus<PS::ECHO_FORM>();
                statuses["FREE_ATTACK_POWER"]   = p.getStatus<PS::FREE_ATTACK_POWER>();
                statuses["REBOUND"]             = p.getStatus<PS::REBOUND>();
                statuses["MANTRA"]              = p.getStatus<PS::MANTRA>();
                statuses["ACCURACY"]            = p.getStatus<PS::ACCURACY>();
                statuses["AFTER_IMAGE"]         = p.getStatus<PS::AFTER_IMAGE>();
                statuses["BATTLE_HYMN"]         = p.getStatus<PS::BATTLE_HYMN>();
                statuses["BRUTALITY"]           = p.getStatus<PS::BRUTALITY>();
                statuses["BURST"]               = p.getStatus<PS::BURST>();
                statuses["COMBUST"]             = p.getStatus<PS::COMBUST>();
                statuses["CREATIVE_AI"]         = p.getStatus<PS::CREATIVE_AI>();
                statuses["DARK_EMBRACE"]        = p.getStatus<PS::DARK_EMBRACE>();
                statuses["DEMON_FORM"]          = p.getStatus<PS::DEMON_FORM>();
                statuses["DEVA"]                = p.getStatus<PS::DEVA>();
                statuses["DEVOTION"]            = p.getStatus<PS::DEVOTION>();
                statuses["DRAW_CARD_NEXT_TURN"] = p.getStatus<PS::DRAW_CARD_NEXT_TURN>();
                statuses["ENERGIZED"]           = p.getStatus<PS::ENERGIZED>();
                statuses["ENVENOM"]             = p.getStatus<PS::ENVENOM>();
                statuses["ESTABLISHMENT"]       = p.getStatus<PS::ESTABLISHMENT>();
                statuses["EVOLVE"]              = p.getStatus<PS::EVOLVE>();
                statuses["FEEL_NO_PAIN"]        = p.getStatus<PS::FEEL_NO_PAIN>();
                statuses["FIRE_BREATHING"]      = p.getStatus<PS::FIRE_BREATHING>();
                statuses["FLAME_BARRIER"]       = p.getStatus<PS::FLAME_BARRIER>();
                statuses["FOCUS"]               = p.focus;
                statuses["FORESIGHT"]           = p.getStatus<PS::FORESIGHT>();
                statuses["HELLO_WORLD"]         = p.getStatus<PS::HELLO_WORLD>();
                statuses["INFINITE_BLADES"]     = p.getStatus<PS::INFINITE_BLADES>();
                statuses["JUGGERNAUT"]          = p.getStatus<PS::JUGGERNAUT>();
                statuses["LIKE_WATER"]          = p.getStatus<PS::LIKE_WATER>();
                statuses["LOOP"]                = p.getStatus<PS::LOOP>();
                statuses["MAGNETISM"]           = p.getStatus<PS::MAGNETISM>();
                statuses["MAYHEM"]              = p.getStatus<PS::MAYHEM>();
                statuses["METALLICIZE"]         = p.getStatus<PS::METALLICIZE>();
                statuses["NEXT_TURN_BLOCK"]     = p.getStatus<PS::NEXT_TURN_BLOCK>();
                statuses["NOXIOUS_FUMES"]       = p.getStatus<PS::NOXIOUS_FUMES>();
                statuses["OMEGA"]               = p.getStatus<PS::OMEGA>();
                statuses["PANACHE"]             = p.getStatus<PS::PANACHE>();
                statuses["PHANTASMAL"]          = p.getStatus<PS::PHANTASMAL>();
                statuses["PLATED_ARMOR"]        = p.getStatus<PS::PLATED_ARMOR>();
                statuses["RAGE"]                = p.getStatus<PS::RAGE>();
                statuses["REGEN"]               = p.getStatus<PS::REGEN>();
                statuses["RITUAL"]              = p.getStatus<PS::RITUAL>();
                statuses["RUPTURE"]             = p.getStatus<PS::RUPTURE>();
                statuses["SADISTIC"]            = p.getStatus<PS::SADISTIC>();
                statuses["STATIC_DISCHARGE"]    = p.getStatus<PS::STATIC_DISCHARGE>();
                statuses["THORNS"]              = p.getStatus<PS::THORNS>();
                statuses["THOUSAND_CUTS"]       = p.getStatus<PS::THOUSAND_CUTS>();
                statuses["TOOLS_OF_THE_TRADE"]  = p.getStatus<PS::TOOLS_OF_THE_TRADE>();
                statuses["VIGOR"]               = p.getStatus<PS::VIGOR>();
                statuses["WAVE_OF_THE_HAND"]    = p.getStatus<PS::WAVE_OF_THE_HAND>();
                statuses["EQUILIBRIUM"]         = p.getStatus<PS::EQUILIBRIUM>();
                statuses["ARTIFACT"]            = p.artifact;
                statuses["DEXTERITY"]           = p.dexterity;
                statuses["STRENGTH"]            = p.strength;
                statuses["THE_BOMB"]            = static_cast<int>(p.bomb3);
                d["statuses"] = statuses;

                // --- Relic presence bits (two uint64 packed as Python ints) ---
                d["relic_bits0"] = static_cast<unsigned long long>(p.relicBits0);
                d["relic_bits1"] = static_cast<unsigned long long>(p.relicBits1);

                // --- Potions ---
                d["potion_count"]    = bc.potionCount;
                d["potion_capacity"] = bc.potionCapacity;
                pybind11::list potions;
                for (int i = 0; i < bc.potionCapacity && i < 5; ++i) {
                    potions.append(std::string(potionEnumNames[static_cast<int>(bc.potions[i])]));
                }
                d["potions"] = potions;

                // --- Card piles ---
                auto make_card_list = [&card_to_dict](auto &arr, int count) {
                    pybind11::list lst;
                    for (int i = 0; i < count; ++i) {
                        lst.append(card_to_dict(arr[i]));
                    }
                    return lst;
                };
                d["hand"] = make_card_list(bc.cards.hand, bc.cards.cardsInHand);

                pybind11::list draw;
                for (const auto &c : bc.cards.drawPile) draw.append(card_to_dict(c));
                d["draw_pile"] = draw;

                pybind11::list discard;
                for (const auto &c : bc.cards.discardPile) discard.append(card_to_dict(c));
                d["discard_pile"] = discard;

                pybind11::list exhaust;
                for (const auto &c : bc.cards.exhaustPile) exhaust.append(card_to_dict(c));
                d["exhaust_pile"] = exhaust;

                // --- Monsters ---
                d["monster_count"] = bc.monsters.monsterCount;
                pybind11::list monsters;
                for (int i = 0; i < bc.monsters.monsterCount && i < 5; ++i) {
                    monsters.append(monster_to_dict(bc.monsters.arr[i], bc));
                }
                d["monsters"] = monsters;

                // --- Combat metadata extras ---
                d["stolen_gold_check"]      = bc.requiresStolenGoldCheck();
                d["last_targeted_monster"]  = static_cast<int>(bc.player.lastTargetedMonster);
                // Stasis cards (Bronze Automaton): convert CardId to string ID, "NONE" for INVALID
                {
                    const auto &sc = bc.cards.stasisCards;
                    d["stasis_card_0"] = (sc[0].id != CardId::INVALID)
                        ? std::string(cardStringIds[static_cast<int>(sc[0].id)])
                        : std::string("NONE");
                    d["stasis_card_1"] = (sc[1].id != CardId::INVALID)
                        ? std::string(cardStringIds[static_cast<int>(sc[1].id)])
                        : std::string("NONE");
                }

                // --- Combat metadata ---
                d["encounter"]   = std::string(monsterEncounterEnumNames[static_cast<int>(bc.encounter)]);
                d["input_state"] = static_cast<int>(bc.inputState);  // adapter maps int -> name
                d["is_over"]     = (bc.outcome != Outcome::UNDECIDED);
                d["outcome"]     = static_cast<int>(bc.outcome);

                // --- Card select info ---
                {
                    const auto &csi = bc.cardSelectInfo;
                    pybind11::dict csd;
                    csd["task"]          = std::string(cardSelectTaskStrings[static_cast<int>(csi.cardSelectTask)]);
                    csd["pick_count"]    = csi.pickCount;
                    csd["can_pick_zero"] = csi.canPickZero;
                    csd["can_pick_any"]  = csi.canPickAnyNumber;
                    pybind11::list cscards;
                    for (int i = 0; i < 3; ++i) {
                        cscards.append(std::string(cardStringIds[static_cast<int>(csi.cards[i])]));
                    }
                    csd["cards"] = cscards;
                    d["card_select_info"] = csd;
                }

                return d;
            },
            "Extract complete battle state as a Python dict for use by LightspeedCombatAdapter");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// os.add_dll_directory("C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin")


