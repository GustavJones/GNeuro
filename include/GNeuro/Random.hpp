#pragma once
#include "GNeuro/Type.hpp"
#include <random>

namespace GNeuro {
inline DECIMAL_T Random() {
  std::random_device rd;
  std::default_random_engine e(rd());
  std::uniform_real_distribution<DECIMAL_T> distribution(-1.0, 1.0);
  return distribution(e);
}

inline DECIMAL_T Random(const DECIMAL_T _bot, const DECIMAL_T _top) {
  std::random_device rd;
  std::default_random_engine e(rd());
  std::uniform_real_distribution<DECIMAL_T> distribution(_bot, _top);
  return distribution(e);
}
} // namespace GNeuro
