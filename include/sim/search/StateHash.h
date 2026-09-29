// The bc.state_hash() FNV-1a hash (moved verbatim out of the pybind lambda so the C++ tree's
// extract cache keys on exactly what the Python search's `extract_cache` keys on; PLAN-perf-cpp-search C.4e).
#ifndef STS_LIGHTSPEED_STATEHASH_H
#define STS_LIGHTSPEED_STATEHASH_H
#include <cstdint>
#include "combat/BattleContext.h"
namespace sts::search {
    std::uint64_t pyStateHash(const BattleContext &bc);
}
#endif
