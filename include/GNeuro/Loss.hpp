/*
 * This file defines the loss functions that are used in the GNeuro::Network
 * class.
 */

#pragma once
#include <stdexcept>
#include <string>
#include <cmath>
#include "GMath/Matrix.hpp"

namespace GNeuro {
/*
 * Returns the difference between the output and it's expected value.
 * When in derivative mode it returns 1 (the derivative of x - _).
 *
 * Return "Error" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> Error(const GMath::Matrix<value_t> &_out, const GMath::Matrix<value_t> &_expected, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::Error";

	if (!_out.IsRowMatrix()) {
		throw std::runtime_error("Outputs are not a row matrix.");
	}

	if (!_expected.IsRowMatrix()) {
		throw std::runtime_error("Expected outputs are not a row matrix");
	}

	if (_out.Shape().Columns != _expected.Shape().Columns) {
		throw std::runtime_error("Outputs and expected outputs count do not match");
	}

	GMath::Matrix<value_t> output(_out.Shape());
	for (GMath::size_t i = 0; i < _out.Shape().Columns; i++) {
		if (_derived) {
			output[0][i] = 1;
		} else {
			output[0][i] =  (_out[0][i] - _expected[0][i]);
		}
	}

	return output;
}

/*
 * Returns the difference between the the expected value and the output.
 * When in derivative mode it returns -1 (the derivative of _ - x).
 *
 * Return "NegativeError" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> NegativeError(const GMath::Matrix<value_t> &_out, const GMath::Matrix<value_t> &_expected, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::NegativeError";

	if (!_out.IsRowMatrix()) {
		throw std::runtime_error("Outputs are not a row matrix.");
	}

	if (!_expected.IsRowMatrix()) {
		throw std::runtime_error("Expected outputs are not a row matrix");
	}

	if (_out.Shape().Columns != _expected.Shape().Columns) {
		throw std::runtime_error("Outputs and expected outputs count do not match");
	}

	GMath::Matrix<value_t> output(_out.Shape());
	for (GMath::size_t i = 0; i < _out.Shape().Columns; i++) {
		if (_derived) {
			output[0][i] = -1;
		} else {
			output[0][i] =  (_expected[0][i] - _out[0][i]);
		}
	}

	return output;
}

/*
 * Returns the Squared Error between the output and the expected output.
 * When in derivative mode it returns the derivative of the Squared Error with the _out as it's input.
 *
 * Return "SquaredError" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> SquaredError(const GMath::Matrix<value_t> &_out, const GMath::Matrix<value_t> &_expected, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::SquaredError";

	if (!_out.IsRowMatrix()) {
		throw std::runtime_error("Outputs are not a row matrix.");
	}

	if (!_expected.IsRowMatrix()) {
		throw std::runtime_error("Expected outputs are not a row matrix");
	}

	if (_out.Shape().Columns != _expected.Shape().Columns) {
		throw std::runtime_error("Outputs and expected outputs count do not match");
	}

  std::string tmp;
  if (_derived) {
		auto error = Error(_out, _expected, false, tmp);
		auto errorSlope = Error(_out, _expected, true, tmp);

		for (GMath::size_t i = 0; i < error.Shape().Columns; i++) {
			error[0][i] *= errorSlope[0][i] * 2;
		}

		return error;
  } else {
		auto output = Error(_out, _expected, false, tmp);

		for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
			output[0][i] = std::pow(output[0][i], 2);
		}

    return output;
  }
}

/*
 * Returns the Squared Negative Error between the output and the expected output.
 * When in derivative mode it returns the derivative of the Squared Negative Error with the _out as it's input.
 *
 * Return "SquaredNegativeError" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> SquaredNegativeError(const GMath::Matrix<value_t> &_out, const GMath::Matrix<value_t> &_expected, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::SquaredNegativeError";

	if (!_out.IsRowMatrix()) {
		throw std::runtime_error("Outputs are not a row matrix.");
	}

	if (!_expected.IsRowMatrix()) {
		throw std::runtime_error("Expected outputs are not a row matrix");
	}

	if (_out.Shape().Columns != _expected.Shape().Columns) {
		throw std::runtime_error("Outputs and expected outputs count do not match");
	}

  std::string tmp;
  if (_derived) {
		auto error = NegativeError(_out, _expected, false, tmp);
		auto errorSlope = NegativeError(_out, _expected, true, tmp);

		for (GMath::size_t i = 0; i < error.Shape().Columns; i++) {
			error[0][i] *= errorSlope[0][i] * 2;
		}

    return error;
  } else {
		auto output = NegativeError(_out, _expected, false, tmp);

		for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
			output[0][i] = std::pow(output[0][i], 2);
		}

    return output;
  }
}
} // namespace GNeuro
