#pragma once
#include "FunctionType.hpp"
#include "Layer.hpp"

namespace GNeuro {
template <typename value_t> class Model {
public:
	class ModelError : public std::runtime_error {
	public:
		ModelError(const std::string &_methodName, const std::string &_errorString) : std::runtime_error("(GNeuro::Model): " + _methodName + " - " + _errorString) {}
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
		void SetLayers(const GMath::size_t &_layerCount) {
			m_structure.Resize(_layerCount);
		}

		/*
		 * Returns the amount of layers in the structure.
		 */
		GMath::size_t Layers() const {
			return m_structure.Size();
		}

		/*
		 * Returns the amount of neurons in a layer, in the structure.
		 */
		GMath::size_t Neurons(const GMath::size_t &_layerIndex) const {
			const static std::string METHOD_NAME = "ModelStructure::Neurons()";

			if (_layerIndex < 0 || _layerIndex >= Layers()) {
				throw ModelError(METHOD_NAME, "Layer index out of bounds.");
			}

			return m_structure[_layerIndex].Size();
		}

		/*
		 * Set a layer in the structure.
		 */
		void SetLayer(const GMath::DynamicArray<value_t> &_layer, const GMath::size_t &_layerIndex) {
			const static std::string METHOD_NAME = "ModelStructure::SetLayer()";

			if (_layerIndex < 0 || _layerIndex >= Layers()) {
				throw ModelError(METHOD_NAME, "Layer index out of bounds.");
			}

			m_structure[_layerIndex] = _layer;
		}

		/*
		 * Retrieve a layer from the structure.
		 */
		const GMath::DynamicArray<value_t> &GetLayer(const GMath::size_t &_layerIndex) const {
			const static std::string METHOD_NAME = "ModelStructure::GetLayer()";

			if (_layerIndex < 0 || _layerIndex >= Layers()) {
				throw ModelError(METHOD_NAME, "Layer index out of bounds.");
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
	void _Check() {
		const static std::string METHOD_NAME = "_Check()";

		if (m_layers.Size() < 1) {
			throw ModelError(METHOD_NAME, "Empty model.");
		}

		if (!m_lossFunction) {
			throw ModelError(METHOD_NAME, "Model has no loss function.");
		}
	}

	// General slope calculations
	// O = A(b + (i1 * w1) + (i2 * w2) + ... + (in * wn))
	// dO/dwn = A'(b + (i1 * w1) + (i2 * w2) + ... + (in * wn)) * in
	// dO/db = A'(b + (i1 * w1) + (i2 * w2) + ... + (in * wn))
	// dO/din = A'(b + (i1 * w1) + (i2 * w2) + ... + (in * wn)) * wn

	// Last layer
	// L = L(O1 + O2 + ... + On)
	// dL/dwnm = L'(O1 + O2 + ... + On) * dOn/dbn * inm
	
	// e.g
	// 1 input, 2 neurons, 3 neurons - Layer network
	//
	// L = L(O11 + O12 + O13)
	// O1n = A1n(b1n + (O21 * w1n1) + (O22 * w1n2))
	// O2m = A2m(b2m + (i1 * w2m1) + (i1 * w2m2))
	// 
	// L = L(
	//   A11(b11 + 
	//     (A21(b21 + (i1 * w211) + (i2 * w212)) * w111) + 
	//     (A22(b22 + (i1 * w221) + (i2 * w222)) * w112)
	//   ) + 
	//
	//   A12(b12 + 
	//     (A21(b21 + (i1 * w211) + (i2 * w212)) * w121) + 
	//     (A22(b22 + (i1 * w221) + (i2 * w222)) * w122)
	//   ) + 
	//
	//   A13(b13 + 
	//     (A21(b21 + (i1 * w211) + (i2 * w212)) * w131) + 
	//     (A22(b22 + (i1 * w221) + (i2 * w222)) * w132)
	//   )
	// )
	//
	// dL/di1 = 
	//   L'(
	//     A11(b11 + 
	//       (A21(b21 + (i1 * w211) + (i2 * w212)) * w111) + 
	//       (A22(b22 + (i1 * w221) + (i2 * w222)) * w112)
	//     ) + 
	//  
	//     A12(b12 + 
	//       (A21(b21 + (i1 * w211) + (i2 * w212)) * w121) + 
	//       (A22(b22 + (i1 * w221) + (i2 * w222)) * w122)
	//     ) + 
	//  
	//     A13(b13 + 
	//       (A21(b21 + (i1 * w211) + (i2 * w212)) * w131) + 
	//       (A22(b22 + (i1 * w221) + (i2 * w222)) * w132)
	//     )
	//   ) *
	//       A11'(b11 + 
	//         (A21(b21 + (i1 * w211) + (i2 * w212)) * w111) + 
	//         (A22(b22 + (i1 * w221) + (i2 * w222)) * w112)
	//       ) *
	//         A21'(b21 + (i1 * w211) + (i2 * w212)) * w111 *
	//				   w211
	//         + 
	//         A22'(b22 + (i1 * w221) + (i2 * w222)) * w112 *
	//           w221
	//       + 
	//    
	//       A12'(b12 + 
	//         (A21(b21 + (i1 * w211) + (i2 * w212)) * w121) + 
	//         (A22(b22 + (i1 * w221) + (i2 * w222)) * w122)
	//       ) * 
	//         A21'(b21 + (i1 * w211) + (i2 * w212)) * w121 *
	//				   w211
	//         +
	//         A22'(b22 + (i1 * w221) + (i2 * w222)) * w122 *
	//           w221
	//       + 
	//    
	//       A13'(b13 + 
	//         (A21(b21 + (i1 * w211) + (i2 * w212)) * w131) + 
	//         (A22(b22 + (i1 * w221) + (i2 * w222)) * w132)
	//       ) *
	//         A21'(b21 + (i1 * w211) + (i2 * w212)) * w131 *
	//           w211
	//         + 
	//         A22'(b22 + (i1 * w221) + (i2 * w222)) * w132 *
	//					 w221
	

	// General slope calculations
	// O = A(b + (i1 * w1) + (i2 * w2) + ... + (in * wn))
	// dO/dwn = A'(b + (i1 * w1) + (i2 * w2) + ... + (in * wn)) * in
	// dO/db = A'(b + (i1 * w1) + (i2 * w2) + ... + (in * wn))
	// dO/din = A'(b + (i1 * w1) + (i2 * w2) + ... + (in * wn)) * wn

	/*
	 * Calculate the loss slope for a certain neuron in the last layer.
	 * IE - dLoss/dOutput_n
	 */
	GMath::Matrix<value_t> _LossSlope(const GMath::Matrix<value_t> &_modelOutputs, const GMath::Matrix<value_t> &_expectedOutputs) {
		const static std::string METHOD_NAME = "_LossSlope()";

		try {
			_Check();
		} catch (...) {
			throw ModelError(METHOD_NAME, "Model checks failed.");
		}

		if (!_modelOutputs.IsRowMatrix()) {
			throw ModelError(METHOD_NAME, "Outputs are not a row matrix.");
		}

		if (!_expectedOutputs.IsRowMatrix()) {
			throw ModelError(METHOD_NAME, "Expected outputs are not a row matrix.");
		}

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
	ModelStructure _ModelStructure() const {
		ModelStructure output;
		output.Setup(m_layers);

		return output;
	}

	/*
	 * Populate activated and unactivated structures of the model.
	 */
	void _OutputStructures(const GMath::Matrix<value_t> &_inputs, ModelStructure &_unactivated, ModelStructure &_activated) {
		const static std::string METHOD_NAME = "_UnactivatedStructure()";
		
		if (_unactivated.Layers() != m_layers.Size() || _activated.Layers() != m_layers.Size()) {
			throw ModelError(METHOD_NAME, "Structure shape does not match model.");
		}

		for (GMath::size_t l = 0; l < m_layers.Size(); l++) {
			if (_unactivated.Neurons(l) != m_layers[l].Size() || _activated.Neurons(l) != m_layers[l].Size()) {
				throw ModelError(METHOD_NAME, "Structure shape does not match model.");
			}
		}

		Layer<value_t> &firstLayer = m_layers[0];
		GMath::Matrix<value_t> unactivated = firstLayer.CalculateLayer(_inputs, false);
		GMath::Matrix<value_t> activated = firstLayer.CalculateLayer(unactivated);
		_unactivated.SetLayer(unactivated[0], 0);
		_activated.SetLayer(activated[0], 0);

		for (GMath::size_t l = 1; l < m_layers.Size(); l++) {
			Layer<value_t> &layer = m_layers[l];
			unactivated = layer.CalculateLayer(activated, false);
			activated = layer.CalculateLayer(unactivated);

			_unactivated.SetLayer(unactivated[0], l);
			_activated.SetLayer(activated[0], l);
		}
	}

	/*
	 * Populate a model structure with unactivated gradients of the loss to unactivated outputs.
	 * IE dLoss/dUnactivatedOutput
	 */
	void _PopulateUnactivatedGradientStructure(ModelStructure &_structure, const ModelStructure &_activatedOutputs, const ModelStructure &_unactivatedOutputs, const GMath::Matrix<value_t> &_expectedOutputs) {
		// Last layer
		GMath::Matrix<value_t> gradients = _LossSlope(_activatedOutputs.GetLayer(m_layers.Size() - 1), _expectedOutputs);
		GMath::Matrix<value_t> activationGradients = m_layers[m_layers.Size() - 1].ActivationSlopes(_unactivatedOutputs.GetLayer(m_layers.Size() - 1));

		for (GMath::size_t n = 0; n < m_layers[m_layers.Size() - 1].Size(); n++) {
			gradients[0][n] *= activationGradients[0][n];
		}

		_structure.SetLayer(gradients[0], _structure.Layers() - 1);

		// All other layers
		for (int64_t l = static_cast<int64_t>(m_layers.Size()) - 2; l >= 0; l--) {
			gradients = m_layers[l].ActivationSlopes(_unactivatedOutputs.GetLayer(l));;

			for (GMath::size_t i = 0; i < m_layers[l].Size(); i++) {
				value_t previousLayerInputSlopeSum = 0;
				for (GMath::size_t n = 0; n < m_layers[l + 1].Size(); n++) {
					previousLayerInputSlopeSum += m_layers[l + 1].InputSlope(_structure.GetLayer(l + 1)[n], n, i);
				}
				
				gradients[0][i] *= previousLayerInputSlopeSum;
			}

			_structure.SetLayer(gradients[0], l);
		}
	}

public:
	/*
	 * Set the loss function to use for model.
	 */
	void SetLossFunction(const typename GNeuro::FunctionType<value_t>::loss_t _lossFunction) {
		m_lossFunction = _lossFunction;
	}

	/*
	 * Add a layer to the model.
	 */
	void AddLayer(const GMath::size_t &_neuronCount, const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {
		const static std::string METHOD_NAME = "AddLayer()";

		try {
			GNeuro::Layer<value_t> layer;
			layer.Set(_neuronCount, _activationFunction);
			m_layers.PushBack(layer);
		} catch (...) {
			throw ModelError(METHOD_NAME, "Failed to add layer.");
		}

	}

	/*
	 * Remove a layer from the model.
	 */
	void RemoveLayer(const GMath::size_t _index) {
		const static std::string METHOD_NAME = "RemoveLayer()";

		try {
			m_layers.Erase(_index);
		} catch (...) {
			throw ModelError(METHOD_NAME, "Failed to remove layer.");
		}
	}

	/*
	 * Fit layers to input count.
	 */
	void Fit(const GMath::size_t &_inputCount) {
		const static std::string METHOD_NAME = "Fit()";

		if (_inputCount <= 0) {
			throw ModelError(METHOD_NAME, "Cannot fit model to input count.");
		}

		m_layers[0].Fit(_inputCount);

		for (GMath::size_t i = 1; i < m_layers.Size(); i++) {
			m_layers[i].Fit(m_layers[i - 1].Size());
		}
	}

	/*
	 * Randomize layer parameters.
	 */
	void Randomize() {
		for (GMath::size_t i = 0; i < m_layers.Size(); i++) {
			m_layers[i].Randomize();
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
	 * Get the mean loss of an input output pair.
	 */
	value_t MeanLoss(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expected) {
		std::string _;
		GMath::Matrix<value_t> losses = m_lossFunction(FeedForward(_inputs), _expected, false, _);

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
	GMath::Matrix<value_t> FeedForward(const GMath::Matrix<value_t> &_inputs) {
		const static std::string METHOD_NAME = "FeedForward()";

		if (!_inputs.IsRowMatrix()) {
			throw ModelError(METHOD_NAME, "Inputs are not a row matrix.");
		}

		try {
			_Check();
		} catch (ModelError &_modelError) {
			throw ModelError(METHOD_NAME, "Model checks failed - \n" + (std::string)_modelError.what());
		}

		GMath::Matrix<value_t> outputs;

		try {
			Layer<value_t> &firstLayer = m_layers[0];
			outputs = firstLayer.CalculateLayer(_inputs, true);

			for (GMath::size_t i = 1; i < m_layers.Size(); i++) {
				Layer<value_t> &layer = m_layers[i];
				outputs = layer.CalculateLayer(outputs, true);
			}
		} catch (...) {
			throw ModelError(METHOD_NAME, "Error running inputs through model.");	
		}

		return outputs;
	}

	/*
	 * Modify model parameters to better suit expected outputs.
	 */
	void BackPropagate(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expectedOutputs, const value_t &_learningRate) {
		const static std::string METHOD_NAME = "BackPropagate()";

		if (!_inputs.IsRowMatrix()) {
			throw ModelError(METHOD_NAME, "Inputs are not a row matrix.");
		}

		if (!_expectedOutputs.IsRowMatrix()) {
			throw ModelError(METHOD_NAME, "Expected outputs are not a row matrix.");
		}

		try {
			_Check();
		} catch (...) {
			throw ModelError(METHOD_NAME, "Model checks failed.");
		}

		ModelStructure unactivatedOutputs = _ModelStructure(), activatedOutputs = _ModelStructure(), gradients = _ModelStructure();
		_OutputStructures(_inputs, unactivatedOutputs, activatedOutputs);

		_PopulateUnactivatedGradientStructure(gradients, activatedOutputs, unactivatedOutputs, _expectedOutputs);

		for (GMath::size_t l = 0; l < m_layers.Size(); l++) {
			Layer<value_t> &currentLayer = m_layers[l];

			for (GMath::size_t n = 0; n < currentLayer.Size(); n++) {
				auto biasSlope = currentLayer.BiasSlope(gradients.GetLayer(l)[n]);
				currentLayer.SetBias(currentLayer.GetBias(n) - biasSlope * _learningRate, n);

				if (l == 0) {
					for (GMath::size_t i = 0; i < currentLayer.Inputs(); i++) {
						auto weightSlope = currentLayer.WeightSlope(gradients.GetLayer(l)[n], _inputs, i);
						currentLayer.SetWeight(currentLayer.GetWeight(n, i) - weightSlope * _learningRate, n, i);
					}
				}
				else {
					for (GMath::size_t i = 0; i < currentLayer.Inputs(); i++) {
						auto weightSlope = currentLayer.WeightSlope(gradients.GetLayer(l)[n], activatedOutputs.GetLayer(l - 1), i);
						currentLayer.SetWeight(currentLayer.GetWeight(n, i) - weightSlope * _learningRate, n, i);
					}
				}
			}
		}
	}
};
} // namespace GNeuro
