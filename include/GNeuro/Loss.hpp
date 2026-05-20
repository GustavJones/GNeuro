/*
 * This file defines the loss functions that are used in the GNeuro::Network
 * class.
 */

#pragma once
#include <string>
#include <cmath>

namespace GNeuro {
/*
 * Returns the difference between the output and it's expected value.
 * When in derivative mode it returns 1 (the derivative of x - _).
 *
 * Return "Error" -> _funcName
 */
template<typename value_t>
inline value_t Error(value_t _out, value_t _expected, bool _derived, std::string &_funcName) {
  _funcName = "Error";

  if (_derived) {
    return 1;
  } else {
    return (_out - _expected);
  }
}

/*
 * Returns the difference between the the expected value and the output.
 * When in derivative mode it returns -1 (the derivative of _ - x).
 *
 * Return "NegativeError" -> _funcName
 */
template<typename value_t>
inline value_t NegativeError(value_t _out, value_t _expected, bool _derived, std::string &_funcName) {
  _funcName = "NegativeError";

  if (_derived) {
    return -1;
  } else {
    return (_expected - _out);
  }
}

/*
 * Returns the Squared Error between the output and the expected output.
 * When in derivative mode it returns the derivative of the Squared Error with the _out as it's input.
 *
 * Return "SquaredError" -> _funcName
 */
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

/*
 * Returns the Squared Negative Error between the output and the expected output.
 * When in derivative mode it returns the derivative of the Squared Negative Error with the _out as it's input.
 *
 * Return "SquaredNegativeError" -> _funcName
 */
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
