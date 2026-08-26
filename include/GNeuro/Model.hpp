#pragma once
#include "FunctionType.hpp"
#include "Layer.hpp"

namespace GNeuro {
template <typename value_t> class Model {
public:
	class ModelError : public std::runtime_error {
	public:
		ModelError(const std::string &_errorString) : std::runtime_error("(GNeuro::Model): " + _errorString) {}
	};

	class ModelStructure {
	private:
		GMath::DynamicArray<GMath::DynamicArray<value_t>> m_structure;

	public:
		/*
		 * Set the dimentions of the structure.
		 */
		void Setup(const GMath::DynamicArray<Layer<value_t>> &_layers) {
			m_structure.Resize(_layers.Size());

			for (GMath::size_t l = 0; l < m_structure.Size(); l++) {
				m_structure[l].Resize(_layers[l].Size());
			}
		}

		/*
		 * Set the amount of layers in the structure.
		 */
		void SetLayerCount(const GMath::size_t &_layerCount) {
			m_structure.Resize(_layerCount);
		}

		/*
		 * Returns the amount of layers in the structure.
		 */
		[[nodiscard]]
		GMath::size_t GetLayerCount() const {
			return m_structure.Size();
		}

		/*
		 * Returns the amount of neurons in a layer, in the structure.
		 */
		[[nodiscard]]
		[[deprecated]]
		GMath::size_t Neurons(const GMath::size_t &_layerIndex) const {

			if (_layerIndex < 0 || _layerIndex >= GetLayerCount()) {
				throw ModelError("Layer index out of bounds.");
			}

			return m_structure[_layerIndex].Size();
		}

		/*
		 * Set a layer in the structure.
		 */
		[[deprecated]]
		void SetLayer(const GMath::DynamicArray<value_t> &_layer, const GMath::size_t &_layerIndex) {

			if (_layerIndex < 0 || _layerIndex >= GetLayerCount()) {
				throw ModelError("Layer index out of bounds.");
			}

			m_structure[_layerIndex] = _layer;
		}

		/*
		 * Retrieve a layer from the structure.
		 */
		[[nodiscard]]
		[[deprecated]]
		const GMath::DynamicArray<value_t> &GetLayer(const GMath::size_t &_layerIndex) const {
			if (_layerIndex < 0 || _layerIndex >= GetLayerCount()) {
				throw ModelError("Layer index out of bounds.");
			}

			return m_structure[_layerIndex];
		}

		/*
		 * Access a layer from the structure.
		 */
		[[nodiscard]]
		GMath::DynamicArray<value_t> &operator[](const GMath::size_t &_layerIndex) {
			if (_layerIndex < 0 || _layerIndex >= GetLayerCount()) {
				throw ModelError("Layer index out of bounds.");
			}

			return m_structure[_layerIndex];
		}


		/*
		 * Access a layer from the structure.
		 */
		[[nodiscard]]
		const GMath::DynamicArray<value_t> &operator[](const GMath::size_t &_layerIndex) const {
			if (_layerIndex < 0 || _layerIndex >= GetLayerCount()) {
				throw ModelError("Layer index out of bounds.");
			}

			return m_structure[_layerIndex];
		}
	};

private:
	GMath::DynamicArray<Layer<value_t>> m_layers;
	typename GNeuro::FunctionType<value_t>::loss_t m_lossFunction = nullptr;

	/*
	 * Check for a valid model object.
	 */
	void _Check() const {

		if (m_layers.Size() < 1) {
			throw ModelError("Empty model.");
		}

		if (!m_lossFunction) {
			throw ModelError("Model has no loss function.");
		}
	}

	/*
	 * Calculate the loss slope for a certain neuron in the last layer.
	 * IE - dLoss/dOutput_n
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> _LossSlope(const GMath::Matrix<value_t> &_modelOutputs, const GMath::Matrix<value_t> &_expectedOutputs) const noexcept {
		std::string _;
		GMath::Matrix<value_t> lossSlopes = m_lossFunction(_modelOutputs, _expectedOutputs, true, _);

		for (GMath::size_t i = 0; i < lossSlopes.Shape().Columns; i++) {
			lossSlopes[0][i] *= ((value_t)1.0/(value_t)lossSlopes.Shape().Columns);
		};

		return lossSlopes;
	}

	/*
	 * Return an empty model structure for this object.
	 */
	[[nodiscard]]
	ModelStructure _ModelStructure() const noexcept {
		ModelStructure output;
		output.Setup(m_layers);

		return output;
	}

	/*
	 * Populate activated and unactivated structures of the model.
	 */
	void _OutputStructures(const GMath::Matrix<value_t> &_inputs, ModelStructure &_unactivated, ModelStructure &_activated) const noexcept {
		const Layer<value_t> &firstLayer = m_layers[0];
		GMath::Matrix<value_t> unactivated = firstLayer.CalculateUnactivated(_inputs);
		GMath::Matrix<value_t> activated = firstLayer.CalculateActivated(unactivated);
		_unactivated[0] = unactivated[0];
		_activated[0] = activated[0];

		for (GMath::size_t l = 1; l < m_layers.Size(); l++) {
			const Layer<value_t> &layer = m_layers[l];
			unactivated = layer.CalculateUnactivated(activated);
			activated = layer.CalculateActivated(unactivated);

			_unactivated[l] = unactivated[0];
			_activated[l] = activated[0];
		}
	}

	/*
	 * Populate a model structure with unactivated gradients of the loss to unactivated outputs.
	 * IE dLoss/dUnactivatedOutput
	 */
	void _PopulateUnactivatedGradientStructure(ModelStructure &_structure, const ModelStructure &_activatedOutputs, const ModelStructure &_unactivatedOutputs, const GMath::Matrix<value_t> &_expectedOutputs) const noexcept {
		// Last layer
		GMath::Matrix<value_t> gradients = _LossSlope(_activatedOutputs[m_layers.Size() - 1], _expectedOutputs);
		GMath::Matrix<value_t> activationGradients = m_layers[m_layers.Size() - 1].ActivationSlopes(_unactivatedOutputs[m_layers.Size() - 1]);

		for (GMath::size_t n = 0; n < m_layers[m_layers.Size() - 1].Size(); n++) {
			gradients[0][n] *= activationGradients[0][n];
		}

		_structure[_structure.GetLayerCount() - 1] = gradients[0];

		// All other layers
		for (int64_t l = static_cast<int64_t>(m_layers.Size()) - 2; l >= 0; l--) {
			gradients = m_layers[l].ActivationSlopes(_unactivatedOutputs[l]);;

			for (GMath::size_t i = 0; i < m_layers[l].Size(); i++) {
				value_t previousLayerInputSlopeSum = 0;
				for (GMath::size_t n = 0; n < m_layers[l + 1].Size(); n++) {
					previousLayerInputSlopeSum += m_layers[l + 1].InputSlope(_structure[l + 1][n], n, i);
				}
				
				gradients[0][i] *= previousLayerInputSlopeSum;
			}

			_structure[l] = gradients[0];
		}
	}

public:
	/*
	 * Set the loss function to use for model.
	 */
	void SetLossFunction(const typename GNeuro::FunctionType<value_t>::loss_t _lossFunction) {
		if (!_lossFunction) {
			throw ModelError("No loss function given.");
		}

		m_lossFunction = _lossFunction;
	}

	/*
	 * Add a layer to the model.
	 */
	void AddLayer(const GMath::size_t &_neuronCount, const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {
		if (_neuronCount < 1) {
			throw ModelError("Invalid neuron count.");
		}

		if (!_activationFunction) {
			throw ModelError("No activation function given.");
		}

		GNeuro::Layer<value_t> layer;
		layer.Set(_neuronCount, _activationFunction);
		m_layers.PushBack(layer);
	}

	/*
	 * Remove a layer from the model.
	 */
	void RemoveLayer(const GMath::size_t _index) {
		if (_index < 0 || _index >= m_layers.Size()) {
			throw ModelError("Index out of bounds.");
		}

		try {
			m_layers.Erase(_index);
		} catch (...) {
			throw ModelError("Failed to remove layer.");
		}
	}

	/*
	 * Fit layers to input count.
	 */
	void Fit(const GMath::size_t &_inputCount) {
		if (_inputCount < 1) {
			throw ModelError("Cannot fit model to input count.");
		}

		m_layers[0].Fit(_inputCount);

		for (GMath::size_t i = 1; i < m_layers.Size(); i++) {
			m_layers[i].Fit(m_layers[i - 1].Size());
		}
	}

	/*
	 * Reset the model to invalid state.
	 */
	void Reset() {
		m_layers.Clear();
		m_lossFunction = nullptr;
	}

	/*
	 * Randomize layer parameters.
	 */
	void Randomize() {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		for (GMath::size_t i = 0; i < m_layers.Size(); i++) {
			m_layers[i].Randomize();
		}
	}

	/*
	 * Get the amount of layers in the model.
	 */
	[[nodiscard]]
	GMath::size_t Layers() const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		return m_layers.Size();
	}

	/*
	 * Get the amount of inputs the model supports.
	 */
	[[nodiscard]]
	GMath::size_t Inputs() const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		return m_layers[0].Inputs();
	}

	/*
	 * Get the amount of outputs the model supports.
	 */
	GMath::size_t Outputs() const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		return m_layers[Layers() - 1].Size();
	}

	/*
	 * Get the mean loss of an input output pair.
	 */
	value_t MeanLoss(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expected) const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		if (!_inputs.IsRowMatrix()) {
			throw ModelError("Inputs is not a row matrix.");
		}

		if (_inputs.Shape().Columns != Inputs()) {
			throw ModelError("Inputs size does not match inputs amount.");
		}

		if (!_expected.IsRowMatrix()) {
			throw ModelError("Expected outputs is not a row matrix.");
		}

		if (_expected.Shape().Columns != Outputs()) {
			throw ModelError("Expected outputs size does not match model outputs size.");
		}

		GMath::Matrix<value_t> outputs;

		try {
			outputs = FeedForward(_inputs);
		} catch (...) {
			throw ModelError("Failed to get outputs from inputs.");
		}

		std::string _;
		GMath::Matrix<value_t> losses = m_lossFunction(outputs, _expected, false, _);

		value_t mean = 0;
		for (GMath::size_t i = 0; i < losses[0].Size(); i++) {
			mean += losses[0][i];
		}

		mean /= losses[0].Size();
		return mean;
	}

	/*
	 * Pass an input through the entire network's parameters.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> FeedForward(const GMath::Matrix<value_t> &_inputs) const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		if (!_inputs.IsRowMatrix()) {
			throw ModelError("Inputs are not a row matrix.");
		}

		if (_inputs.Shape().Columns != Inputs()) {
			throw ModelError("Inputs size does not match inputs amount.");
		}

		GMath::Matrix<value_t> outputs;
		const Layer<value_t> &firstLayer = m_layers[0];
		outputs = firstLayer.CalculateActivated(firstLayer.CalculateUnactivated(_inputs));

		for (GMath::size_t i = 1; i < m_layers.Size(); i++) {
			const Layer<value_t> &layer = m_layers[i];
			outputs = layer.CalculateActivated(layer.CalculateUnactivated(outputs));
		}

		return outputs;
	}

	/*
	 * Modify model parameters to better suit expected outputs.
	 */
	void BackPropagate(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expectedOutputs, const value_t &_learningRate) {
		try {
			_Check();
		} catch (...) {
			throw ModelError("Model checks failed.");
		}

		if (!_inputs.IsRowMatrix()) {
			throw ModelError("Inputs are not a row matrix.");
		}

		if (_inputs.Shape().Columns != Inputs()) {
			throw ModelError("Inputs size does not match inputs amount.");
		}

		if (!_expectedOutputs.IsRowMatrix()) {
			throw ModelError("Expected outputs are not a row matrix.");
		}

		if (_expectedOutputs.Shape().Columns != Outputs()) {
			throw ModelError("Expected outputs size does not match model output size.");
		}

		if (_learningRate <= 0) {
			throw ModelError("Learning rate has to be > 0.");
		}

		ModelStructure unactivatedOutputs = _ModelStructure(), activatedOutputs = _ModelStructure(), gradients = _ModelStructure();
		_OutputStructures(_inputs, unactivatedOutputs, activatedOutputs);
		_PopulateUnactivatedGradientStructure(gradients, activatedOutputs, unactivatedOutputs, _expectedOutputs);

		for (GMath::size_t l = 0; l < m_layers.Size(); l++) {
			Layer<value_t> &currentLayer = m_layers[l];

			for (GMath::size_t n = 0; n < currentLayer.Size(); n++) {
				auto biasSlope = currentLayer.BiasSlope(gradients[l][n]);
				currentLayer.SetBias(currentLayer.GetBias(n) - biasSlope * _learningRate, n);

				if (l == 0) {
					for (GMath::size_t i = 0; i < currentLayer.Inputs(); i++) {
						auto weightSlope = currentLayer.WeightSlope(gradients[l][n], _inputs, i);
						currentLayer.SetWeight(currentLayer.GetWeight(n, i) - weightSlope * _learningRate, n, i);
					}
				}
				else {
					for (GMath::size_t i = 0; i < currentLayer.Inputs(); i++) {
						auto weightSlope = currentLayer.WeightSlope(gradients[l][n], activatedOutputs[l - 1], i);
						currentLayer.SetWeight(currentLayer.GetWeight(n, i) - weightSlope * _learningRate, n, i);
					}
				}
			}
		}
	}
};
} // namespace GNeuro
