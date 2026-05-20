/*
 * This file implements a wrapper around the C++ random tools.
 */

#pragma once
#include <random>

namespace GNeuro {

/*
 * Generate a random value between _bot and _top.
 */
template<typename value_t>
inline value_t Random(const value_t _bot, const value_t _top) {
  std::random_device rd;
  std::default_random_engine e(rd());
  std::uniform_real_distribution<value_t> distribution(_bot, _top);
  return distribution(e);
}
} // namespace GNeuro
