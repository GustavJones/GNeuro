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
		LayerError(const std::string &_errorString) : std::runtime_error("(GNeuro::Layer): " + _errorString) {}
	};

private:
	GMath::Matrix<value_t> m_weights;
	GMath::DynamicArray<value_t> m_biases;
	typename FunctionType<value_t>::activation_t m_activationFunction;

	/*
	 * Check if the layer's weights and biases create a valid layer.
	 * Throws and exception if errors were found.
	 */
	void _Check() const {

		GMath::MatrixShape weightsShape = m_weights.Shape();
		GMath::size_t biasesCount = m_biases.Size();

		if (biasesCount <= 0) {
			throw LayerError("Invalid layer. Not enough neurons.");
		}

		if (weightsShape.Rows != biasesCount) {
			throw LayerError("Bias count does not match available weight sets.");
		}

		GMath::size_t inputCount = m_weights[0].Size();
		if (inputCount <= 0) {
			throw LayerError("Layer doesn't support any input.");
		}

		if (!m_activationFunction) {
				throw LayerError("Layer has an unset activation function.");
		}

		for (GMath::size_t i = 1; i < biasesCount; i++) {
			const GMath::DynamicArray<value_t> &row = m_weights[i];

			if (row.Size() != inputCount) {
				throw LayerError("Weight set " + std::to_string(i) + " doesn't match input count of " + std::to_string(inputCount));
			}
		}
	}

	/*
	 * Calculate the unactivated output values of _inputs when run through the layer.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> _CalculateUnactivated(const GMath::Matrix<value_t> &_inputs) const noexcept {
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
	GMath::Matrix<value_t> _CalculateActivated(const GMath::Matrix<value_t> &_unactivatedOutputs) const noexcept {
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
	[[nodiscard]]
	GMath::size_t Size() const {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		return m_biases.Size();
	}

	/*
	 * Resize the layer.
	 */
	void Resize(const GMath::size_t &_neuronCount) {
		if (_neuronCount < 1) {
			throw LayerError("Invalid neuron count.");
		}

		m_biases.Resize(_neuronCount);
		m_weights.Reshape({m_biases.Size(), m_weights.Shape().Columns});

		m_weights.Zero();
		for (GMath::size_t i = 0; i < m_biases.Size(); i++) {
			m_biases[i] = 0;
		}
	}

	/*
	 * Get the amount of inputs supported by the layer.
	 */
	[[nodiscard]]
	GMath::size_t Inputs() const {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		return m_weights[0].Size();
	}

	/*
	 * Setup layer to a specific size and activation.
	 */
	void Set(const GMath::size_t &_neuronCount, const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {

		if (_neuronCount < 1) {
			throw LayerError("Invalid neuron count.");
		}

		if (!_activationFunction) {
			throw LayerError("Invalid activation function.");
		}

		Resize(_neuronCount);
		SetActivation(_activationFunction);
	}

	/*
	 * Reset the layer to invalid state.
	 */
	void Reset() {
		m_biases.Resize(0);
		m_weights.Reshape({0, 0});
		m_activationFunction = nullptr;
	}

	/*
	 * Configure layer for an input amount.
	 */
	void Fit(const GMath::size_t &_inputCount) {

		if (_inputCount < 1) {
			throw LayerError("Invalid input count.");
		}

		m_weights.Reshape({m_weights.Shape().Rows, _inputCount});
		m_weights.Zero();
	}

	/*
	 * Set parameters to random values.
	 */
	void Randomize() {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

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
	GMath::Matrix<value_t> CalculateUnactivated(const GMath::Matrix<value_t> &_inputs) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_inputs.IsRowMatrix()) {
			throw LayerError("Inputs are not a row matrix.");
		}

		if (_inputs.Shape().Columns != Inputs()) {
			throw LayerError("Layer does not support input size. Layer requires " + std::to_string(m_weights[0].Size()) + " inputs.");
		}

		auto unactivated = _CalculateUnactivated(_inputs);
		return unactivated;
	}

	/*
	 * Calculate the activated output value of _inputs when run through the layer.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> CalculateActivated(const GMath::Matrix<value_t> &_unactivatedOutput) const {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_unactivatedOutput.IsRowMatrix()) {
			throw LayerError("Unactivated outputs is not a row matrix.");
		}

		if (_unactivatedOutput.Shape().Columns != Size()) {
			throw LayerError("Unactivated outputs size does not match layer size.");
		}

		return _CalculateActivated(_unactivatedOutput);
	}

	[[nodiscard]]
	GMath::Matrix<value_t> Calculate(const GMath::Matrix<value_t> &_inputs) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_inputs.IsRowMatrix()) {
			throw LayerError("Inputs is not a row matrix.");		
		}

		if (_inputs.Shape().Columns != Inputs()) {
			throw LayerError("Inputs amount not supported by layer.");
		}

		try {
		  CalculateActivated(CalculateUnactivated(_inputs));
		} catch (...) {
			throw LayerError("Failed to calculate layer.");
		}
	}

	/*
	 * Get the bias of a certain neuron.
	 */
	[[nodiscard]]
	value_t GetBias(const GMath::size_t &_neuronIndex) const {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError("Bias index out of bounds.");
		}

		return m_biases[_neuronIndex];
	}

	/*
	 * Set the bias of a certain neuron.
	 */
	void SetBias(const value_t _value, const GMath::size_t &_neuronIndex) {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError("Bias index out of bounds.");
		}

		m_biases[_neuronIndex] = _value;
	}

	/*
	 * Get the weight of a certain neuron and input.
	 */
	[[nodiscard]]
	value_t GetWeight(const GMath::size_t &_neuronIndex, const GMath::size_t &_inputIndex) const {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError("Weight index out of bounds.");
		}

		if (_inputIndex < 0 || _inputIndex >= Inputs()) {
			throw LayerError("Weight index out of bounds.");
		}

		return m_weights[_neuronIndex][_inputIndex];
	}

	/*
	 * Set the weight of a certain neuron and input.
	 */
	void SetWeight(const value_t _value, const GMath::size_t &_neuronIndex, const GMath::size_t &_inputIndex) {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError("Weight index out of bounds.");
		}

		if (_inputIndex < 0 || _inputIndex >= Inputs()) {
			throw LayerError("Weight index out of bounds.");
		}

		m_weights[_neuronIndex][_inputIndex] = _value;
	}

	/*
	 * Get the activation function of a certain neuron.
	 */
	[[nodiscard]]
	typename GNeuro::FunctionType<value_t>::activation_t GetActivation() const {

		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		return m_activationFunction;
	}

	void SetActivation(const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {
		if (!_activationFunction) {
			throw LayerError("No activation function given.");
		}

		m_activationFunction = _activationFunction;
	}

	/*
	 * Get the activation slope contribution to the activated output.
	 * IE - dOutput/dActivation
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> ActivationSlopes(const GMath::Matrix<value_t> &_unactivatedOutputs) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_unactivatedOutputs.IsRowMatrix()) {
			throw LayerError("Unactivated outputs is not a row matrix.");
		}

		if (_unactivatedOutputs.Shape().Columns != Size()) {
			throw LayerError("Unactivated outputs size does not match layer size.");
		}

		try {
			typename GNeuro::FunctionType<value_t>::activation_t activation = GetActivation();

			std::string _;
			GMath::Matrix<value_t> activatedDerivatives = activation(_unactivatedOutputs, true, _);

			return activatedDerivatives;
		} catch (LayerError &_layerError) {
			throw LayerError("Error getting activation slope - \n" + (std::string)_layerError.what());
		}
	}

	/*
	 * Get the activation slope contribution to the activated output.
	 * IE - dOutput/dActivation
	 */
	[[nodiscard]]
	[[deprecated]] // TODO Optimize to only calculate the needed value instead of all of the neuron outputs.
	value_t ActivationSlope(const GMath::Matrix<value_t> &_unactivatedOutputs, const GMath::size_t &_neuronIndex) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_unactivatedOutputs.IsRowMatrix()) {
			throw LayerError("Unactivated outputs is not a row matrix.");
		}

		if (_unactivatedOutputs.Shape().Columns != Size()) {
			throw LayerError("Unactivated outputs size does not match layer size.");
		}

		try {
			return ActivationSlopes(_unactivatedOutputs)[_neuronIndex];
		} catch (...) {
			throw LayerError("Failed to calculate activation slopes.");
		}
	}

	/*
	 * Get the bias slope contributions to the activated outputs.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> BiasSlopes(const GMath::Matrix<value_t> &_unactivatedSlopes) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_unactivatedSlopes.IsRowMatrix()) {
			throw LayerError("Unactivated slopes is not a row matrix.");
		}

		if (_unactivatedSlopes.Shape().Columns != Size()) {
			throw LayerError("Unactivated slopes size does not match layer size.");
		}

		return _unactivatedSlopes;
	}

	/*
	 * Get the bias slope contribution to the activated output.
	 * IE - dOutput/dBias
	 */
	[[nodiscard]]
	value_t BiasSlope(const value_t &_unactivatedSlope) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		return _unactivatedSlope;
	}


	/*
	 * Get the weight slope contributions to the activated outputs.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> WeightSlopes(const GMath::Matrix<value_t> &_unactivatedSlopes, const GMath::Matrix<value_t> &_layerInputs) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_unactivatedSlopes.IsRowMatrix()) {
			throw LayerError("Unactivated slopes is not a row matrix.");
		}

		if (_unactivatedSlopes.Shape().Columns != Size()) {
			throw LayerError("Unactivated slopes is not the size of the layer.");
		}

		if (!_layerInputs.IsRowMatrix()) {
			throw LayerError("Layer inputs is not a row matrix.");
		}

		if (_layerInputs.Shape().Columns != Inputs()) {
			throw LayerError("Layer inputs size does not match input count.");
		}

		GMath::Matrix<value_t> weightSlopes;

		try {
			for (GMath::size_t n = 0; n < Size(); n++) {
				GMath::DynamicArray<value_t> weightBatch;

				for (GMath::size_t w = 0; w < _layerInputs.Shape().Columns; w++) {
					weightBatch.PushBack(WeightSlope(_unactivatedSlopes[0][n], _layerInputs, w));
				}

				weightSlopes.AppendRow(weightBatch);
			}
		} catch (...) {
			throw LayerError("Failed to get weight slopes.");
		}

		return weightSlopes;
	}

	/*
	 * Get the weight slope contribution to the activated output.
	 * IE - dOutput/dWeight
	 */
	[[nodiscard]]
	value_t WeightSlope(const value_t &_unactivatedSlope, const GMath::Matrix<value_t> &_layerInputs, const GMath::size_t &_weightIndex) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (!_layerInputs.IsRowMatrix()) {
			throw LayerError("Layer inputs is not a row matrix.");
		}

		if (_layerInputs.Shape().Columns != Inputs()) {
			throw LayerError("Layer inputs size does not match input count.");
		}

		if (_weightIndex < 0 || _weightIndex >= Inputs()) {
			throw LayerError("Weight index out of bounds.");
		}

		return _unactivatedSlope * _layerInputs[0][_weightIndex];
	}

	/*
	 * Get the input slope contribution to the activated output.
	 * IE - dOutput/dInput
	 */
	[[nodiscard]]
	value_t InputSlope(const value_t &_unactivatedSlope, const GMath::size_t &_neuronIndex, const GMath::size_t &_inputIndex) const {
		try {
			_Check();
		} catch (...) {
			throw LayerError("Layer has an unknown shape.");
		}

		if (_neuronIndex < 0 || _neuronIndex >= Size()) {
			throw LayerError("Neuron index out of bounds.");
		}

		if (_inputIndex < 0 || _inputIndex >= Inputs()) {
			throw LayerError("Input index out of bounds.");
		}

		try {
			return _unactivatedSlope * GetWeight(_neuronIndex, _inputIndex);
		} catch (LayerError &_layerError) {
			throw LayerError("Error getting input slope.");
		}
	}

	/*
	 * Get the input slope contributions the the activated outputs.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> InputSlopes(const GMath::Matrix<value_t> &_unactivatedSlopes) const {
		GMath::Matrix<value_t> inputSlopes;

		if (!_unactivatedSlopes.IsRowMatrix()) {
			throw LayerError("Unactivated slopes is not a row matrix.");
		}

		if (_unactivatedSlopes.Shape().Columns != Size()) {
			throw LayerError("Unactivated slopes is not the size of the layer.");
		}

		try {
			for (GMath::size_t n = 0; n < Size(); n++) {
				GMath::DynamicArray<value_t> inputBatch;

				for (GMath::size_t i = 0; i < Inputs(); i++) {
					inputBatch.PushBack(InputSlope(_unactivatedSlopes[0][n], n, i));
				}

				inputSlopes.AppendRow(inputBatch);
			}

			// return _unactivatedSlope * GetWeight(_neuronIndex, _inputIndex);
		} catch (LayerError &_layerError) {
			throw LayerError("Error getting input slope.");
		}

		return inputSlopes;
	}

};
} // namespace GNeuro
