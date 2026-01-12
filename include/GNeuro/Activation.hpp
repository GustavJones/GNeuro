#pragma once
#include "GNeuro/Type.hpp"
#include <string>
#include <utility>
#include <vector>

namespace GNeuro {
inline DECIMAL_T None(DECIMAL_T _in, bool _derived) {
  if (_derived) {
    return 1;
  } else {
    return _in;
  }
}

inline DECIMAL_T Sigmoid(DECIMAL_T _in, bool _derived) {
  if (_derived) {
    return Sigmoid(_in, false) * (1 - Sigmoid(_in, false));
  } else {
    return 1 / (1 + std::exp(-_in));
  }
}

inline DECIMAL_T ReLu(DECIMAL_T _in, bool _derived) {
  if (_derived) {
    return (_in >= 0) ? 1 : 0;
  } else {
    return (_in > 0) ? _in : 0;
  }
}

inline DECIMAL_T LeakyReLu(DECIMAL_T _in, bool _derived) {
  const DECIMAL_T SLOPE = 0.01;

  if (_derived) {
    return (_in >= 0) ? 1 : SLOPE;
  } else {
    return (_in > 0) ? _in : SLOPE * _in;
  }
}

inline DECIMAL_T TanH(DECIMAL_T _in, bool _derived) {
  if (_derived) {
    return std::pow(2 / (std::exp(_in) + std::exp(-_in)), 2);
  } else {
    return (std::exp(_in) - std::exp(-_in)) / (std::exp(_in) + std::exp(-_in));
  }
}

static const std::vector<std::pair<std::string, ACTIVATION_T>> ActivationFunctions = {
  {"None", None},
  {"Sigmoid", Sigmoid},
  {"ReLu", ReLu},
  {"LeakyReLu", LeakyReLu},
  {"TanH", TanH}
};
} // namespace GNeuro
