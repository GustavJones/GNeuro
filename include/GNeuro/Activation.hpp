/*
 * This file defines a few inline activation functions that can be used with the
 * GNeuro library.
 */

#pragma once
#include <string>
#include <cmath>
#include "GMath/Matrix.hpp"

namespace GNeuro {
/*
 * Activation function that does nothing.
 * It returns the value without modification.
 * When in derivative mode it returns 1 (the derivative of x).
 *
 * Returns "None" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> None(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::None";

	if (!_in.IsRowMatrix()) {
		throw std::runtime_error("Inputs are not a row matrix.");
	}

	GMath::Matrix<value_t> output(_in.Shape());

	for (GMath::size_t i = 0; i < _in.Shape().Columns; i++) {
		if (_derived) {
			output[0][i] = 1;
		} else {
			output[0][i] = _in[0][i];
		}
	}

	return output;
}

/*
 * Activation function that returns a Sigmoid activated value.
 * When in derivative mode it returns the derivative of the Sigmoid function with _in as the input.
 *
 * Returns "Sigmoid" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> Sigmoid(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::Sigmoid";

	if (!_in.IsRowMatrix()) {
		throw std::runtime_error("Inputs are not a row matrix.");
	}

	GMath::Matrix<value_t> output(_in.Shape());

  std::string _;
	if (_derived) {
		for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
			auto s = 1 / (1 + std::exp(-_in[0][i]));
			output[0][i] =  s * (1 - s);
		}
	}
	else {
		for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
			output[0][i] = 1 / (1 + std::exp(-_in[0][i]));
		}
	}

	return output;
}

/*
 * Activation function that returns a ReLu activated value.
 * When in derivative mode it returns the derivative of the ReLu function with _in as the input.
 *
 * Returns "ReLu" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> ReLu(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::ReLu";

	if (!_in.IsRowMatrix()) {
		throw std::runtime_error("Inputs are not a row matrix.");
	}

	GMath::Matrix<value_t> output(_in.Shape());

	for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
		if (_derived) {
			output[0][i] = (_in[0][i] >= 0) ? 1 : 0;
		} else {
			output[0][i] = (_in[0][i] >= 0) ? _in[0][i] : 0;
		}
	}

	return output;
}

/*
 * Activation function that returns a Leaky ReLu activated value.
 * When in derivative mode it returns the derivative of the Leaky ReLu function with _in as the input.
 *
 * Returns "LeakyReLu" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> LeakyReLu(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::LeakyReLu";

  const value_t SLOPE = 0.01;

	if (!_in.IsRowMatrix()) {
		throw std::runtime_error("Inputs are not a row matrix.");
	}

	GMath::Matrix<value_t> output(_in.Shape());

	for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
		if (_derived) {
			output[0][i] = (_in[0][i] >= 0) ? 1 : SLOPE;
		} else {
			output[0][i] = (_in[0][i] >= 0) ? _in[0][i] : SLOPE * _in[0][i];
		}
	}

	return output;
}

/*
 * Activation function that returns a TanH activated value.
 * When in derivative mode it returns the derivative of the TanH function with _in as the input.
 *
 * Returns "TanH" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> TanH(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::TanH";

	if (!_in.IsRowMatrix()) {
		throw std::runtime_error("Inputs are not a row matrix.");
	}

	GMath::Matrix<value_t> output(_in.Shape());

	for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
		if (_derived) {
			output[0][i] = std::pow(2 / (std::exp(_in[0][i]) + std::exp(-_in[0][i])), 2);
		} else {
			output[0][i] = (std::exp(_in[0][i]) - std::exp(-_in[0][i])) / (std::exp(_in[0][i]) + std::exp(-_in[0][i]));
		}
	}

	return output;
}

/*
 * Activation fucntion that returns a Softmax activated value.
 * When in derivative mode it returns the derivative of the Softmax function with _in as the input.
 *
 * Return "Softmax" -> _funcName
 */
template<typename value_t>
inline GMath::Matrix<value_t> Softmax(const GMath::Matrix<value_t> &_in, bool _derived, std::string &_funcName) {
  _funcName = "GNeuro::Softmax";

	if (!_in.IsRowMatrix()) {
		throw std::runtime_error("Inputs are not a row matrix.");
	}

	GMath::Matrix<value_t> output = _in;

	value_t max;

	if (output.Shape().Columns > 0) {
		max = output[0][0];

		for (GMath::size_t i = 1; i < output.Shape().Columns; i++) {
			if (output[0][i] > max) {
				max = output[0][i];
			}
		}

		for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
			output[0][i] = output[0][i] - max;
		}


		value_t sum = 0;
		for (GMath::size_t i = 0; i < output.Shape().Columns; i++) {
			output[0][i] = std::exp(output[0][i]);
			sum += output[0][i];
		}

		output = output / sum;
		
	}

	return output;
}
} // namespace GNeuro
