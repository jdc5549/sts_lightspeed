//
// RandomAgent: random action selection for benchmarking.
//
// playoutBattle() is used by the Python bindings (play_battle_random) to run
// a battle with uniformly random action selection.  playout() and
// playoutWithBattles() are declared in the header for completeness but are
// not called from Python and are provided only as linker stubs.
//

#include "sim/RandomAgent.h"
#include "game/GameContext.h"
#include "combat/BattleContext.h"
#include "sim/search/BattleScumSearcher2.h"

using namespace sts;

RandomAgent::RandomAgent(const Rng &rng) : rng(rng) {}

void RandomAgent::playoutBattle(BattleContext &bc) {
    // Reuse BattleScumSearcher2's action enumeration but pick uniformly at random.
    search::BattleScumSearcher2 searcher(bc);
    std::vector<search::Action> actionStack;
    searcher.playoutRandom(bc, actionStack);
}

// Linker stubs — overworld random play is not needed for the Python binding use case.
void RandomAgent::playout(GameContext &gc) {}
void RandomAgent::playout(GameContext &gc, const GameContextPredicate &p) {}
void RandomAgent::playoutWithBattles(GameContext &gc) {}
