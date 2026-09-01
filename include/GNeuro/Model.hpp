#pragma once
#include "FunctionType.hpp"
#include "Layer.hpp"
#include "GParsing/JSON/GParsing-JSON.hpp"

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
			// m_structure.Resize(_layers.Size());
			SetLayerCount(_layers.Size());

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
		[[deprecated("Use operator[].Size() instead.")]]
		GMath::size_t Neurons(const GMath::size_t &_layerIndex) const {

			if (_layerIndex < 0 || _layerIndex >= GetLayerCount()) {
				throw ModelError("Layer index out of bounds.");
			}

			return m_structure[_layerIndex].Size();
		}

		/*
		 * Set a layer in the structure.
		 */
		[[deprecated("Use operator[] instead.")]]
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
		[[deprecated("Use operator[] instead.")]]
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

	class ModelWeightGradients {
	private:
		GMath::DynamicArray<GMath::DynamicArray<GMath::DynamicArray<value_t>>> m_gradients;
	
	public:
		/*
		 * Set the dimentions of the structure.
		 */
		void Setup(const GMath::DynamicArray<Layer<value_t>> &_layers) {
			m_gradients.Resize(_layers.Size());

			for (GMath::size_t l = 0; l < m_gradients.Size(); l++) {
				m_gradients[l].Resize(_layers[l].Size());

				for (GMath::size_t n = 0; n < m_gradients[l].Size(); n++) {
					m_gradients[l][n].Resize(_layers[l].Inputs());
				}
			}
		}

		/*
		 * Access a layer from gradients.
		 */
		GMath::DynamicArray<GMath::DynamicArray<value_t>> &operator[](const GMath::size_t &_index) {
			if (_index < 0 || _index >= m_gradients.Size()) {
				throw ModelError("Index out of bounds.");
			}

			return m_gradients[_index];
		}

		/*
		 * Access a layer from gradients.
		 */
		const GMath::DynamicArray<GMath::DynamicArray<value_t>> &operator[](const GMath::size_t &_index) const {
			if (_index < 0 || _index >= m_gradients.Size()) {
				throw ModelError("Index out of bounds.");
			}

			return m_gradients[_index];
		}

		/*
		 * Get the amount of layers in the structure.
		 */
		[[nodiscard]]
		GMath::size_t Size() const {
			return m_gradients.Size();
		}


		/*
		 * Check if other structure has the same dimentions.
		 */
		[[nodiscard]]
		bool HasSameDimentions(const ModelWeightGradients &_other) const {
			if (Size() != _other.Size()) {
				return false;
			}

			for (GMath::size_t l = 0; l < Size(); l++) {
				if (operator[](l).Size() != _other[l].Size()) {
					return false;
				}

				for (GMath::size_t n = 0; n < operator[](l).Size(); n++) {
					if (operator[](l)[n].Size() != _other[l][n].Size()) {
						return false;
					}
				}
			}

			return true;
		}

		/*
		 * Add weight gradients.
		 */
		[[nodiscard]]
		ModelWeightGradients operator+(const ModelWeightGradients &_add) const {
			if (!HasSameDimentions(_add)) {
				throw ModelError("Add gradients does not have the correct dimentions.");
			}

			ModelWeightGradients output = *this;

			for (GMath::size_t l = 0; l < Size(); l++) {
				for (GMath::size_t n = 0; n < operator[](l).Size(); n++) {
					for (GMath::size_t w = 0; w < operator[](l)[n].Size(); w++) {
						output[l][n][w] += _add[l][n][w];
					}
				}
			}

			return output;
		}

		
		/*
		 * Scalar multiply weight gradients.
		 */
		[[nodiscard]]
		ModelWeightGradients operator*(const value_t &_add) const {
			ModelWeightGradients output = *this;

			for (GMath::size_t l = 0; l < Size(); l++) {
				for (GMath::size_t n = 0; n < operator[](l).Size(); n++) {
					for (GMath::size_t w = 0; w < operator[](l)[n].Size(); w++) {
						output[l][n][w] *= _add;
					}
				}
			}

			return output;
		}
	};

	class ModelBiasGradients {
	private:
		GMath::DynamicArray<GMath::DynamicArray<value_t>> m_gradients;
	public:
		/*
		 * Set the dimentions of the structure.
		 */
		void Setup(const GMath::DynamicArray<Layer<value_t>> &_layers) {
			m_gradients.Resize(_layers.Size());

			for (GMath::size_t l = 0; l < m_gradients.Size(); l++) {
				m_gradients[l].Resize(_layers[l].Size());
			}
		}

		/*
		 * Access a layer from gradients.
		 */
		GMath::DynamicArray<value_t> &operator[](const GMath::size_t &_index) {
			if (_index < 0 || _index >= m_gradients.Size()) {
				throw ModelError("Index out of bounds.");
			}

			return m_gradients[_index];
		}

		/*
		 * Access a layer from gradients.
		 */
		const GMath::DynamicArray<value_t> &operator[](const GMath::size_t &_index) const {
			if (_index < 0 || _index >= m_gradients.Size()) {
				throw ModelError("Index out of bounds.");
			}

			return m_gradients[_index];
		}

		/*
		 * Get the amount of layers in the structure.
		 */
		[[nodiscard]]
		GMath::size_t Size() const {
			return m_gradients.Size();
		}

		/*
		 * Check if other structure has the same dimentions.
		 */
		[[nodiscard]]
		bool HasSameDimentions(const ModelBiasGradients &_other) const {
			if (Size() != _other.Size()) {
				return false;
			}

			for (GMath::size_t l = 0; l < Size(); l++) {
				if (operator[](l).Size() != _other[l].Size()) {
					return false;
				}
			}

			return true;
		}

		/*
		 * Add bias gradients.
		 */
		[[nodiscard]]
		ModelBiasGradients operator+(const ModelBiasGradients &_add) const {
			if (!HasSameDimentions(_add)) {
				throw ModelError("Add gradients does not have the correct dimentions.");
			}

			ModelBiasGradients output = *this;

			for (GMath::size_t l = 0; l < Size(); l++) {
				for (GMath::size_t n = 0; n < operator[](l).Size(); n++) {
					output[l][n] += _add[l][n];
				}
			}

			return output;
		}

		
		/*
		 * Scalar multiply bias gradients.
		 */
		[[nodiscard]]
		ModelBiasGradients operator*(const value_t &_add) const {
			ModelBiasGradients output = *this;

			for (GMath::size_t l = 0; l < Size(); l++) {
				for (GMath::size_t n = 0; n < operator[](l).Size(); n++) {
					output[l][n] *= _add;
				}
			}

			return output;
		}
	};

	struct ModelGradients {
		ModelWeightGradients weights;
		ModelBiasGradients biases;

		/*
		 * Returns the average of all of the gradients.
		 */
		static ModelGradients Average(const GMath::DynamicArray<ModelGradients> &_gradients) {
			if (_gradients.Size() < 1) {
				throw ModelError("No gradients given.");
			}

			for (GMath::size_t g = 0; g < _gradients.Size(); g++) {
				const ModelGradients &currentGradients = _gradients[g];

				if (!currentGradients.StructuresMatch()) {
					throw ModelError("Internal structure dimentions don't match.");
				}
			}

			ModelGradients output = _gradients[0];

			for (GMath::size_t g = 1; g < _gradients.Size(); g++) {
				output = output + _gradients[g];
			}

			output = output * ((value_t)1.00/_gradients.Size());

			return output;
		}

		/*
		 * Setup weights and biases structures.
		 */
		void Setup(const GMath::DynamicArray<Layer<value_t>> &_layers) {
			weights.Setup(_layers);
			biases.Setup(_layers);
		}

		/*
		 * Check whether weights and biases' dimentions match.
		 */
		[[nodiscard]]
		bool StructuresMatch() const {
			if (weights.Size() != biases.Size()) {
				return false;
			}

			for (GMath::size_t l = 0; l < weights.Size(); l++) {
				if (weights[l].Size() != biases[l].Size()) {
					return false;
				}
			}

			return true;
		}

		/*
		 * Get the amount of layers in both structures.
		 */
		[[nodiscard]]
		GMath::size_t Layers() const {
			if (!StructuresMatch()) {
				throw ModelError("Internal structure dimentions don't match.");
			}

			return weights.Size();
		}

		/*
		 * Get the amount of neurons in a structure layer.
		 */
		[[nodiscard]]
		GMath::size_t Neurons(const GMath::size_t _layerIndex) const {
			if (!StructuresMatch()) {
				throw ModelError("Internal structure dimentions don't match.");
			}

			if (_layerIndex < 0 || _layerIndex >= Layers()) {
				throw ModelError("Layer index out of bounds.");
			}

			return weights[_layerIndex].Size();
		}

		/*
		 * Get the amount of weights in the structure neuron.
		 */
		[[nodiscard]]
		GMath::size_t Weights(const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex) const {
			if (!StructuresMatch()) {
				throw ModelError("Internal structure dimentions don't match.");
			}

			if (_layerIndex < 0 || _layerIndex >= Layers()) {
				throw ModelError("Layer index out of bounds.");
			}

			if (_neuronIndex < 0 || _neuronIndex >= Neurons(_layerIndex)) {
				throw ModelError("Neuron index out of bounds.");
			}

			return weights[_layerIndex][_neuronIndex].Size();
		}

		/*
		 * Add gradients.
		 */
		[[nodiscard]]
		ModelGradients operator+(const ModelGradients &_add) const {
			ModelGradients output;
			try {
				output.weights = weights + _add.weights;
				output.biases = biases + _add.biases;
			} catch (...) {
				throw ModelError("Failed to add gradients.");
			}

			return output;
		}


		/*
		 * Scalar multiply gradients.
		 */
		[[nodiscard]]
		ModelGradients operator*(const value_t &_add) const {
			ModelGradients output;
			output.weights = weights * _add;
			output.biases = biases * _add;

			return output;
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
	 * Return an empty gradients structure for this object.
	 */
	ModelGradients _SetupGradients() const {
		ModelGradients output;
		output.Setup(m_layers);

		return output;
	}

	/*
	 * Return an empty model structure for this object.
	 */
	[[nodiscard]]
	ModelStructure _SetupModelStructure() const noexcept {
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


	/*
	 * Version 1 of JSON loading format.
	 */
	void _LoadV1(const GParsing::JSONObject<unsigned char> &_json, const GMath::DynamicArray<typename FunctionType<value_t>::loss_t> &_availableLossFunctions, const GMath::DynamicArray<typename FunctionType<value_t>::activation_t> &_availableActivationFunctions) {
    const auto lossString = _json["loss"].GetString();
    bool found = false;
    for (size_t l = 0; l < _availableLossFunctions.Size(); l++) {
      std::string funcName;
      _availableLossFunctions[l]({0}, {0}, false, funcName);

      if (lossString == funcName) {
				SetLossFunction(_availableLossFunctions[l]);
        found = true;
      }
    }

    if (!found) {
      throw ModelError("Cannot parse loss function string.");
    }

    const auto &weights = _json["weights"].GetArray();
    const auto &biases = _json["biases"].GetArray();
    const auto &activations = _json["activations"].GetArray();

		// Check layer count
    if (weights.GetSize() != biases.GetSize() || weights.GetSize() != activations.GetSize()) {
      throw ModelError("Corrupted model.");
    }

		m_layers.Resize(weights.GetSize());
    for (size_t l = 0; l < weights.GetSize(); l++) {
      auto &jsonWeightsLayer = weights[l].GetArray();
      auto &jsonBiasesLayer = biases[l].GetArray();
      auto activation = activations[l].GetString();

      if (jsonWeightsLayer.GetSize() < 1) {
        continue;
      }

			bool found = false;
			for (size_t a = 0; a < _availableActivationFunctions.Size(); a++) {
				std::string funcName;
				_availableActivationFunctions[a](0, false, funcName);

				if (activation == funcName) {
					m_layers[l].SetActivation(_availableActivationFunctions[a]);
					found = true;
				}
			}

			if (!found) {
				throw ModelError("No activation function found in provided list.");
			}

			m_layers[l].Resize(jsonWeightsLayer.GetSize());
			m_layers[l].Fit(jsonWeightsLayer[0].GetArray().GetSize());

      for (size_t n = 0; n < jsonWeightsLayer.GetSize(); n++) {
        auto &jsonWeight = jsonWeightsLayer[n].GetArray();
        auto bias = jsonBiasesLayer[n].GetNumber();

				m_layers[l].SetBias(bias, n);

        for (size_t w = 0; w < m_layers[l].Inputs(); w++) {
          value_t weight = jsonWeight.GetValue(w).GetNumber();
					m_layers[l].SetWeight(weight, n, w);
        }

      }
    }
  }

public:
	/*
	 * Add the weights and biases of another model.
	 */
	Model operator+(const Model &_addModel) const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		if (m_lossFunction != _addModel.m_lossFunction) {
			throw ModelError("Models use different loss functions");
		}

		if (Inputs() != _addModel.Inputs()) {
			throw ModelError("Model input counts not the same.");
		}

		auto selfDimentions = Dimentions(), otherDimentions = _addModel.Dimentions();
		if (selfDimentions.Size() != otherDimentions.Size()) {
			throw ModelError("Model dimentions do not match.");
		}

		for (GMath::size_t l = 0; l < selfDimentions.Size(); l++) {
			if (selfDimentions[l] != otherDimentions[l]) {
				throw ModelError("Model dimentions do not match.");
			}
		}

		for (GMath::size_t l = 0; l < selfDimentions.Size(); l++) {
			if (m_layers[l].GetActivation() != _addModel.m_layers[l].GetActivation()) {
				throw ModelError("Models use different activation functions.");
			}
		}

		Model output = *this;

		for (GMath::size_t l = 0; l < selfDimentions.Size(); l++) {
			for (GMath::size_t n = 0; n < selfDimentions[l]; n++) {
				auto oldBias = output.m_layers[l].GetBias(n);
				output.m_layers[l].SetBias(oldBias + _addModel.m_layers[l].GetBias(n), n);

				for (GMath::size_t w = 0; w < m_layers[l].Inputs(); w++) {
					auto oldWeight = output.m_layers[l].GetWeight(n, w);
					output.m_layers[l].SetWeight(oldWeight + _addModel.m_layers[l].GetWeight(n, w), n, w);
				}
			}
		}

		return output;
	}

	/*
	 * Subtract the weights and biases of another model.
	 */
	Model operator-(const Model &_subtractModel) const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		if (m_lossFunction != _subtractModel.m_lossFunction) {
			throw ModelError("Models use different loss functions");
		}

		if (Inputs() != _subtractModel.Inputs()) {
			throw ModelError("Model input counts not the same.");
		}

		auto selfDimentions = Dimentions(), otherDimentions = _subtractModel.Dimentions();
		if (selfDimentions.Size() != otherDimentions.Size()) {
			throw ModelError("Model dimentions do not match.");
		}

		for (GMath::size_t l = 0; l < selfDimentions.Size(); l++) {
			if (selfDimentions[l] != otherDimentions[l]) {
				throw ModelError("Model dimentions do not match.");
			}
		}

		for (GMath::size_t l = 0; l < selfDimentions.Size(); l++) {
			if (m_layers[l].GetActivation() != _subtractModel.m_layers[l].GetActivation()) {
				throw ModelError("Models use different activation functions.");
			}
		}

		Model output = *this;

		for (GMath::size_t l = 0; l < selfDimentions.Size(); l++) {
			for (GMath::size_t n = 0; n < selfDimentions[l]; n++) {
				auto oldBias = output.m_layers[l].GetBias(n);
				output.m_layers[l].SetBias(oldBias - _subtractModel.m_layers[l].GetBias(n), n);

				for (GMath::size_t w = 0; w < m_layers[l].Inputs(); w++) {
					auto oldWeight = output.m_layers[l].GetWeight(n, w);
					output.m_layers[l].SetWeight(oldWeight - _subtractModel.m_layers[l].GetWeight(n, w), n, w);
				}
			}
		}

		return output;
	}


	/*
	 * Multiply the weights and biases with a factor.
	 */
	Model operator*(const value_t &_factor) const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		Model output = *this;

		for (GMath::size_t l = 0; l < Layers(); l++) {
			for (GMath::size_t n = 0; n < m_layers[l].Size(); n++) {
				auto oldBias = output.m_layers[l].GetBias(n);
				output.m_layers[l].SetBias(oldBias * _factor, n);

				for (GMath::size_t w = 0; w < m_layers[l].Inputs(); w++) {
					auto oldWeight = output.m_layers[l].GetWeight(n, w);
					output.m_layers[l].SetWeight(oldWeight * _factor, n, w);
				}
			}
		}

		return output;
	}

	/*
	 * Divide the weights and biases with a factor.
	 */
	Model operator/(const value_t &_factor) const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		Model output = *this;

		for (GMath::size_t l = 0; l < Layers(); l++) {
			for (GMath::size_t n = 0; n < m_layers[l].Size(); n++) {
				auto oldBias = output.m_layers[l].GetBias(n);
				output.m_layers[l].SetBias(oldBias / _factor, n);

				for (GMath::size_t w = 0; w < m_layers[l].Inputs(); w++) {
					auto oldWeight = output.m_layers[l].GetWeight(n, w);
					output.m_layers[l].SetWeight(oldWeight / _factor, n, w);
				}
			}
		}

		return output;
	}

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
	 * Get the dimentions of the model.
	 */
	GMath::DynamicArray<GMath::size_t> Dimentions() const {
		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError("Model checks failed - \n" + (std::string)_modelError.what());
		}

		GMath::DynamicArray<GMath::size_t> output;

		for (GMath::size_t l = 0; l < Layers(); l++) {
			output.PushBack(m_layers[l].Size());
		}

		return output;
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
	[[deprecated("Change in API. Use Optimize instead.")]]
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

		ModelStructure unactivatedOutputs = _SetupModelStructure(), activatedOutputs = _SetupModelStructure(), gradients = _SetupModelStructure();
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

	/*
	 * Calculate the model parameter gradients.
	 */
	Model<value_t>::ModelGradients BackPropagate(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expectedOutputs) const {
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

		ModelGradients output = _SetupGradients();
		ModelStructure unactivatedOutputs = _SetupModelStructure(), activatedOutputs = _SetupModelStructure(), gradients = _SetupModelStructure();
		_OutputStructures(_inputs, unactivatedOutputs, activatedOutputs);
		_PopulateUnactivatedGradientStructure(gradients, activatedOutputs, unactivatedOutputs, _expectedOutputs);

		for (GMath::size_t l = 0; l < m_layers.Size(); l++) {
			const Layer<value_t> &currentLayer = m_layers[l];

			for (GMath::size_t n = 0; n < currentLayer.Size(); n++) {
				auto biasSlope = currentLayer.BiasSlope(gradients[l][n]);
				output.biases[l][n] = biasSlope;
				// currentLayer.SetBias(currentLayer.GetBias(n) - biasSlope * _learningRate, n);

				if (l == 0) {
					for (GMath::size_t i = 0; i < currentLayer.Inputs(); i++) {
						auto weightSlope = currentLayer.WeightSlope(gradients[l][n], _inputs, i);
						output.weights[l][n][i] = weightSlope;
						// currentLayer.SetWeight(currentLayer.GetWeight(n, i) - weightSlope * _learningRate, n, i);
					}
				}
				else {
					for (GMath::size_t i = 0; i < currentLayer.Inputs(); i++) {
						auto weightSlope = currentLayer.WeightSlope(gradients[l][n], activatedOutputs[l - 1], i);
						output.weights[l][n][i] = weightSlope;
						// currentLayer.SetWeight(currentLayer.GetWeight(n, i) - weightSlope * _learningRate, n, i);
					}
				}
			}
		}

		return output;
	}

	/*
	 * Use gradients to optimize model.
	 */
	void Optimize(const ModelGradients &_gradients, const value_t &_learningRate) {
		try {
			_Check();
		} catch (...) {
			throw ModelError("Model checks failed.");
		}

		if (_gradients.Layers() != Layers()) {
			throw ModelError("Gradients don't match model layer count.");
		}

		for (GMath::size_t l = 0; l < _gradients.Layers(); l++) {
			if (_gradients.Neurons(l) != m_layers[l].Size()) {
				throw ModelError("Gradients don't match model neuron count.");
			}

			for (GMath::size_t n = 0; n < _gradients.Neurons(l); n++) {
				if (_gradients.Weights(l, n) != m_layers[l].Inputs()) {
					throw ModelError("Gradients don't match model weight count.");
				}
			}
		}

		for (GMath::size_t l = 0; l < _gradients.Layers(); l++) {
			for (GMath::size_t n = 0; n < _gradients.Neurons(l); n++) {
				value_t oldBias = m_layers[l].GetBias(n);
				m_layers[l].SetBias(oldBias - _gradients.biases[l][n], n);

				for (GMath::size_t w = 0; w < _gradients.Weights(l, n); w++) {
					value_t oldWeight = m_layers[l].GetWeight(n, w);
					m_layers[l].SetWeight(oldWeight - _gradients.weights[l][n][w] * _learningRate, n, w);
				}
			}
		}
	}

	/*
	 * Use multiple gradients to optimize model.
	 */
	void Optimize(const GMath::DynamicArray<ModelGradients> &_gradients, const value_t &_learningRate) {
		try {
			_Check();
		} catch (...) {
			throw ModelError("Model checks failed.");
		}
		
		if (_gradients.Size() < 1) {
			throw ModelError("No gradients provided.");
		}
		
		for (GMath::size_t m = 0; m < _gradients.Size(); m++) {
			if (_gradients[m].Layers() != Layers()) {
				throw ModelError("Gradients don't match model layer count.");
			}

			for (GMath::size_t l = 0; l < _gradients[m].Layers(); l++) {
				if (_gradients[m].Neurons(l) != m_layers[l].Size()) {
					throw ModelError("Gradients don't match model neuron count.");
				}

				for (GMath::size_t n = 0; n < _gradients[m].Neurons(l); n++) {
					if (_gradients[m].Weights(l, n) != m_layers[l].Inputs()) {
						throw ModelError("Gradients don't match model weight count.");
					}
				}
			}
		}

		ModelGradients averageGradients = ModelGradients::Average(_gradients);
		Optimize(averageGradients, _learningRate);
	}

	/*
	 * Save the model in a JSON file.
	 */
	void Save(const std::string &_filepath) {
		try {
			_Check();
		} catch (...) {
			throw ModelError("Model checks failed.");
		}

		if (_filepath.empty()) {
			throw ModelError("No filepath given.");
		}

		using serialize_t = unsigned char;
    const std::string MODEL_SAVE_VERSION = "v1";
    GParsing::JSONObject<serialize_t> json;

    // Metadata
    GParsing::JSONObject<serialize_t> metadata;
    metadata.AddMember("version", (GParsing::JSONString<serialize_t>)MODEL_SAVE_VERSION);
    metadata.AddMember("type", (GParsing::JSONString<serialize_t>)"GNeuro::Model");
    json.AddMember("metadata", metadata);

		std::string lossFunctionName;
		m_lossFunction({0}, {0}, false, lossFunctionName);

		json.AddMember("loss", (GParsing::JSONString<serialize_t>)lossFunctionName);


		// Weights
		GParsing::JSONArray<serialize_t> weights;
		for (GMath::size_t l = 0; l < m_layers.Size(); l++) {
			GParsing::JSONArray<serialize_t> layer;

			for (GMath::size_t n = 0; n < m_layers[l].Size(); n++) {
				GParsing::JSONArray<serialize_t> neuron;

				for (GMath::size_t w = 0; w < m_layers[l].Inputs(); w++) {
					neuron.PushValue((GParsing::JSONNumber<serialize_t>)m_layers[l].GetWeight(n, w));
				}  

				layer.PushValue(neuron);
			}

			weights.PushValue(layer);
		}
		json.AddMember("weights", weights);

		// Biases
		GParsing::JSONArray<serialize_t> biases;
		for (size_t l = 0; l < m_layers.Size(); l++) {
			GParsing::JSONArray<serialize_t> layer;

			for (size_t n = 0; n < m_layers[l].Size(); n++) {
				layer.PushValue((GParsing::JSONNumber<serialize_t>)m_layers[l].GetBias(n));
			}

			biases.PushValue(layer);
		}
		json.AddMember("biases", biases);

		// Activation Functions
		GParsing::JSONArray<serialize_t> activations;
		for (size_t l = 0; l < m_layers.Size(); l++) {
			std::string activationFunctionName;
			m_layers[l].GetActivation()({0}, false, activationFunctionName);
			activations.PushValue((GParsing::JSONString<serialize_t>)activationFunctionName);
		}
		json.AddMember("activations", activations);
   
    if (!json.Serialize(_filepath)) {
      throw ModelError("Error serializing model to JSON.");
    }
  }

	/*
	 * Load the model from a JSON file.
	 */
	void Load(const std::string &_filepath, const GMath::DynamicArray<typename FunctionType<value_t>::loss_t> &_availableLossFunctions, const GMath::DynamicArray<typename FunctionType<value_t>::activation_t> &_availableActivationFunctions) {
    GParsing::JSONObject<unsigned char> json;
    if (!json.Parse(_filepath)) {
      throw std::runtime_error("Error parsing model from JSON.");
    }

    const auto &metadataObject = json["metadata"].GetObject();
    const auto &versionString = metadataObject["version"].GetString();
    const auto &typeString = metadataObject["type"].GetString();

    if (typeString != "GNeuro::Model") {
      throw std::runtime_error("Unknown JSON type.");
    }

    if (versionString == "v1") {
      _LoadV1(json, _availableLossFunctions, _availableActivationFunctions);
    }
    else {
      throw std::runtime_error("Unknown model version");
    }
  }
};
} // namespace GNeuro
