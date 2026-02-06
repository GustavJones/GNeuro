#pragma once
#include <string>
#include <cmath>

namespace GNeuro {
template<typename value_t>
inline value_t None(value_t _in, bool _derived, std::string &_funcName) {
  _funcName = "None";

  if (_derived) {
    return 1;
  } else {
    return _in;
  }
}

template<typename value_t>
inline value_t Sigmoid(value_t _in, bool _derived, std::string &_funcName) {
  _funcName = "Sigmoid";

  std::string tmp;
  if (_derived) {
    return Sigmoid(_in, false, tmp) * (1 - Sigmoid(_in, false, tmp));
  } else {
    return 1 / (1 + std::exp(-_in));
  }
}

template<typename value_t>
inline value_t ReLu(value_t _in, bool _derived, std::string &_funcName) {
  _funcName = "ReLu";

  if (_derived) {
    return (_in >= 0) ? 1 : 0;
  } else {
    return (_in > 0) ? _in : 0;
  }
}

template<typename value_t>
inline value_t LeakyReLu(value_t _in, bool _derived, std::string &_funcName) {
  _funcName = "LeakyReLu";

  const value_t SLOPE = 0.01;

  if (_derived) {
    return (_in >= 0) ? 1 : SLOPE;
  } else {
    return (_in > 0) ? _in : SLOPE * _in;
  }
}

template<typename value_t>
inline value_t TanH(value_t _in, bool _derived, std::string &_funcName) {
  _funcName = "TanH";

  if (_derived) {
    return std::pow(2 / (std::exp(_in) + std::exp(-_in)), 2);
  } else {
    return (std::exp(_in) - std::exp(-_in)) / (std::exp(_in) + std::exp(-_in));
  }
}
} // namespace GNeuro
