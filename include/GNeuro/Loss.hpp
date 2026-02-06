#pragma once
#include <string>
#include <cmath>

namespace GNeuro {
template<typename value_t>
inline value_t Error(value_t _out, value_t _expected, bool _derived, std::string &_funcName) {
  _funcName = "Error";

  if (_derived) {
    return 1;
  } else {
    return (_out - _expected);
  }
}

template<typename value_t>
inline value_t NegativeError(value_t _out, value_t _expected, bool _derived, std::string &_funcName) {
  _funcName = "NegativeError";

  if (_derived) {
    return -1;
  } else {
    return (_expected - _out);
  }
}

template<typename value_t>
inline value_t SquaredError(value_t _out, value_t _expected, bool _derived, std::string &_funcName) {
  _funcName = "SquaredError";

  std::string tmp;
  if (_derived) {
    return Error(_out, _expected, true, tmp) * 2 * Error(_out, _expected, false, tmp);
  } else {
    return std::pow(Error(_out, _expected, false, tmp), 2);
  }
}

template<typename value_t>
inline value_t SquaredNegativeError(value_t _out, value_t _expected, bool _derived, std::string &_funcName) {
  _funcName = "SquaredNegativeError";

  std::string tmp;
  if (_derived) {
    return NegativeError(_out, _expected, true, tmp) * 2 * NegativeError(_out, _expected, false, tmp);
  } else {
    return std::pow(NegativeError(_out, _expected, false, tmp), 2);
  }
}
} // namespace GNeuro
