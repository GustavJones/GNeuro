#pragma once
#include "GNeuro/Type.hpp"

namespace GNeuro {
inline DECIMAL_T Error(DECIMAL_T _out, DECIMAL_T _expected, bool _derived) {
  if (_derived) {
    return 1;
  } else {
    return (_out - _expected);
  }
}

inline DECIMAL_T NegativeError(DECIMAL_T _out, DECIMAL_T _expected, bool _derived) {
  if (_derived) {
    return -1;
  } else {
    return (_expected - _out);
  }
}

inline DECIMAL_T SquaredError(DECIMAL_T _out, DECIMAL_T _expected, bool _derived) {
  if (_derived) {
    return Error(_out, _expected, true) * 2 * Error(_out, _expected, false);
  } else {
    return std::pow(Error(_out, _expected, false), 2);
  }
}

inline DECIMAL_T SquaredNegativeError(DECIMAL_T _out, DECIMAL_T _expected, bool _derived) {
  if (_derived) {
    return NegativeError(_out, _expected, true) * 2 * NegativeError(_out, _expected, false);
  } else {
    return std::pow(NegativeError(_out, _expected, false), 2);
  }
}
} // namespace GNeuro
