#pragma once
#include "GMath/Matrix.hpp"
#include "GMath/Random.hpp"
#include "GMath/DynamicArray.hpp"
#include "GNeuro/FunctionType.hpp"
#include <stdexcept>

namespace GNeuro {
/*
 * A layer in a neural network. Used by Model to contain an entire model's parameters.
 */
template <typename value_t> class Layer {
public:
	class LayerError : public std::runtime_error {
	public:
		LayerError(const std::string &_methodName, const std::string &_errorString) : std::runtime_error("(GNeuro::Layer): " + _methodName + " - " + _errorString) {}
	};

private:
	GMath::Matrix<value_t> m_weights;
	GMath::DynamicArray<value_t> m_biases;
	typename FunctionType<value_t>::activation_t m_activationFunction;
	/*
	 * Check if the layer's weights and biases create a valid layer.
	 * Throws and exception if errors were found.
	 */
	void _CheckShape() const {
		const static std::string METHOD_NAME = "_CheckShape()";

		GMath::MatrixShape weightsShape = m_weights.Shape();
		GMath::size_t biasesCount = m_biases.Size();

		if (biasesCount <= 0) {
			throw LayerError(METHOD_NAME, "Invalid layer. Not enough neurons.");
		}

		if (weightsShape.Rows != biasesCount) {
			throw LayerError(METHOD_NAME, "Bias count does not match available weight sets.");
		}

		GMath::size_t inputCount = m_weights[0].Size();
		if (inputCount <= 0) {
			throw LayerError(METHOD_NAME, "Layer doesn't support any input.");
		}

		if (!m_activationFunction) {
				throw LayerError(METHOD_NAME, "Layer has an unset activation function.");
		}

		for (GMath::size_t i = 1; i < biasesCount; i++) {
			const GMath::DynamicArray<value_t> &row = m_weights[i];

			if (row.Size() != inputCount) {
				throw LayerError(METHOD_NAME, "Weight set " + std::to_string(i) + " doesn't match input count of " + std::to_string(inputCount));
			}
		}
	}

	/*
	 * Calculate the unactivated output values of _inputs when run through the layer.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> _CalculateUnactivated(const GMath::Matrix<value_t> &_inputs) const {
		const static std::string METHOD_NAME = "CalculateUnactivated()";

		if (!_inputs.IsRowMatrix()) {
			throw LayerError(METHOD_NAME, "Inputs are not a row matrix.");
		}

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		if (_inputs.Shape().Columns != m_weights[0].Size()) {
			throw LayerError(METHOD_NAME, "Layer does not support input size. Layer requires " + std::to_string(m_weights[0].Size()) + " inputs.");
		}

		GMath::Matrix<value_t> unactivatedOutputs = (_inputs * m_weights.Transpose()) + m_biases;
		return unactivatedOutputs;
		GMath::Matrix<value_t> activatedOutputs { unactivatedOutputs.Shape() };

		std::string _;
		activatedOutputs = GetActivation()(unactivatedOutputs, false, _);

		return activatedOutputs;
	}

	/*
	 * Calculate the activated output values of _inputs when run through the layer.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> _CalculateActivated(const GMath::Matrix<value_t> &_unactivatedOutputs) const {
		const static std::string METHOD_NAME = "CalculateActivated()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		std::string _;
		GMath::Matrix<value_t> activatedOutputs = GetActivation()(_unactivatedOutputs, false, _);
		
		return activatedOutputs;
	}

public:
	Layer() = default;
	Layer(const Layer &_layer) = default;
	Layer(Layer &&_layer) = default;
	Layer& operator=(const Layer &_layer) = default;
	Layer& operator=(Layer &&_layer) = default;
	~Layer() = default;

	/*
	 * Get the amount of neurons in the layer.
	 */
	GMath::size_t Size() const {
		const static std::string METHOD_NAME = "Size()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		return m_biases.Size();
	}

	GMath::size_t Inputs() const {
		const static std::string METHOD_NAME = "Inputs()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		return m_weights[0].Size();
	}

	/*
	 * Setup layer to a specific size and activation.
	 */
	void Set(const GMath::size_t &_neuronCount, const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {
		const static std::string METHOD_NAME = "Set()";

		if (_neuronCount <= 0) {
			throw LayerError(METHOD_NAME, "Invalid neuron count.");
		}

		if (!_activationFunction) {
			throw LayerError(METHOD_NAME, "Invalid activation function.");
		}

		m_biases.Resize(_neuronCount);
		m_weights.Reshape({m_biases.Size(), m_weights.Shape().Columns});

		m_weights.Zero();
		m_activationFunction = _activationFunction;
		for (GMath::size_t i = 0; i < m_biases.Size(); i++) {
			m_biases[i] = 0;
		}
	}

	/*
	 * Configure layer for an input amount.
	 */
	void Fit(const GMath::size_t &_inputCount) {
		const static std::string METHOD_NAME = "Fit()";

		if (_inputCount <= 0) {
			throw LayerError(METHOD_NAME, "Invalid input count.");
		}

		m_weights.Reshape({m_weights.Shape().Rows, _inputCount});
		m_weights.Zero();
	}

	/*
	 * Set parameters to random values.
	 */
	void Randomize() {
		for (GMath::size_t n = 0; n < Size(); n++) {
			m_biases[n] = GMath::Random((value_t)-1, (value_t)1);

			for (GMath::size_t i = 0; i < Inputs(); i++) {
				m_weights[n][i] = GMath::Random((value_t)-1, (value_t)1);
			}
		}
	}

	/*
	 * Calculate the output value of _inputs when run through the layer.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> CalculateLayer(const GMath::Matrix<value_t> &_inputs, const bool _activated) const {
		const static std::string METHOD_NAME = "CalculateLayer()";
		try {
			auto unactivated = _CalculateUnactivated(_inputs);

			if (_activated) return _CalculateActivated(unactivated);
			else return unactivated;
		} catch (LayerError &_layerError) {
			throw LayerError(METHOD_NAME, "Failed to calculate outputs - " + (std::string)_layerError.what());
		}	
	}

	/*
	 * Calculate the activated output value of _inputs when run through the layer.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> CalculateLayer(const GMath::Matrix<value_t> &_unactivatedOutput) const {
		const static std::string METHOD_NAME = "CalculateLayer()";
		try {
			return _CalculateActivated(_unactivatedOutput);
		} catch (LayerError &_layerError) {
			throw LayerError(METHOD_NAME, "Failed to calculate outputs - " + (std::string)_layerError.what());
		}
	}

	/*
	 * Get the bias of a certain neuron.
	 */
	[[nodiscard]]
	value_t GetBias(const GMath::size_t &_neuronIndex) const {
		const static std::string METHOD_NAME = "GetBias()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError(METHOD_NAME, "Bias index out of bounds.");
		}

		return m_biases[_neuronIndex];
	}

	/*
	 * Set the bias of a certain neuron.
	 */
	void SetBias(const value_t _value, const GMath::size_t &_neuronIndex) {
		const static std::string METHOD_NAME = "SetBias()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError(METHOD_NAME, "Bias index out of bounds.");
		}

		m_biases[_neuronIndex] = _value;
	}

	/*
	 * Get the weight of a certain neuron and input.
	 */
	[[nodiscard]]
	value_t GetWeight(const GMath::size_t &_neuronIndex, const GMath::size_t &_inputIndex) const {
		const static std::string METHOD_NAME = "GetWeight()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError(METHOD_NAME, "Weight index out of bounds.");
		}

		if (_inputIndex < 0 || _inputIndex >= Inputs()) {
			throw LayerError(METHOD_NAME, "Weight index out of bounds.");
		}

		return m_weights[_neuronIndex][_inputIndex];
	}

	/*
	 * Set the weight of a certain neuron and input.
	 */
	void SetWeight(const value_t _value, const GMath::size_t &_neuronIndex, const GMath::size_t &_inputIndex) {
		const static std::string METHOD_NAME = "SetWeight()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError(METHOD_NAME, "Weight index out of bounds.");
		}

		if (_inputIndex < 0 || _inputIndex >= Inputs()) {
			throw LayerError(METHOD_NAME, "Weight index out of bounds.");
		}

		m_weights[_neuronIndex][_inputIndex] = _value;
	}

	/*
	 * Get the activation function of a certain neuron.
	 */
	[[nodiscard]]
	typename GNeuro::FunctionType<value_t>::activation_t GetActivation() const {
		const static std::string METHOD_NAME = "GetActivation()";

		try {
			_CheckShape();
		} catch (...) {
			throw LayerError(METHOD_NAME, "Layer has an unknown shape.");
		}

		return m_activationFunction;
	}

	/*
	 * Get the activation slope contribution to the activated output.
	 * IE - dOutput/dActivation
	 */
	GMath::Matrix<value_t> ActivationSlopes(const GMath::Matrix<value_t> &_unactivatedOutputs) const {
		const static std::string METHOD_NAME = "ActivationSlope()";

		try {
			typename GNeuro::FunctionType<value_t>::activation_t activation = GetActivation();

			std::string _;
			GMath::Matrix<value_t> activatedDerivatives = activation(_unactivatedOutputs, true, _);

			return activatedDerivatives;
		} catch (LayerError &_layerError) {
			throw LayerError(METHOD_NAME, "Error getting activation slope - \n" + (std::string)_layerError.what());
		}
	}

	/*
	 * Get the activation slope contribution to the activated output.
	 * IE - dOutput/dActivation
	 */
	value_t ActivationSlope(const GMath::Matrix<value_t> &_unactivatedOutputs, const GMath::size_t &_neuronIndex) const {
		// const static std::string METHOD_NAME = "ActivationSlope()";
		//
		// try {
		// 	typename GNeuro::FunctionType<value_t>::activation_t activation = GetActivation();
		//
		// 	std::string _;
		// 	GMath::Matrix<value_t> activatedDerivatives = activation(_unactivatedOutputs, true, _);
		//
		// 	return activatedDerivatives[0][_neuronIndex];
		// } catch (LayerError &_layerError) {
		// 	throw LayerError(METHOD_NAME, "Error getting activation slope - \n" + (std::string)_layerError.what());
		// }

		// TODO Optimize to only calculate the needed value instead of all of the neuron outputs.
		return ActivationSlopes(_unactivatedOutputs)[_neuronIndex];
	}

	/*
	 * Get the bias slope contribution to the activated output.
	 * IE - dOutput/dBias
	 */
	value_t BiasSlope(const value_t &_unactivatedSlope) const {
		return _unactivatedSlope;
	}

	/*
	 * Get the weight slope contribution to the activated output.
	 * IE - dOutput/dWeight
	 */
	value_t WeightSlope(const value_t &_unactivatedSlope, const GMath::Matrix<value_t> &_layerInputs, const GMath::size_t &_weightIndex) const {
		const static std::string METHOD_NAME = "WeightSlope()";

		try {
			return _unactivatedSlope * _layerInputs[0][_weightIndex];
		} catch (LayerError &_layerError) {
			throw LayerError(METHOD_NAME, "Error getting weight slope - \n" + (std::string)_layerError.what());
		}
	}

	/*
	 * Get the input slope contribution to the activated output.
	 * IE - dOutput/dInput
	 */
	value_t InputSlope(const value_t &_unactivatedSlope, const GMath::size_t &_neuronIndex, const GMath::size_t &_inputIndex) const {
		const static std::string METHOD_NAME = "InputSlope()";

		try {
			return _unactivatedSlope * GetWeight(_neuronIndex, _inputIndex);
		} catch (LayerError &_layerError) {
			throw LayerError(METHOD_NAME, "Error getting input slope - \n" + (std::string)_layerError.what());
		}
	}
};
} // namespace GNeuro
