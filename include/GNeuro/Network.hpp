#pragma once
#include "GNeuro/Model.hpp"
#include <stdexcept>
#include <atomic>

namespace GNeuro {

/*
 * An object to create a Neural Network. Uses different GNeuro::Layers to define
 * a network that can be created, trained, used for calculations, cleared, saved
 * and loaded.
 */
template <typename value_t> class Network {
public:
	class NetworkError : public std::runtime_error {
	public:
		NetworkError(const std::string &_msg) : std::runtime_error("(GNeuro::Network): " + _msg) {}
	};

private:
	GNeuro::Model<value_t> m_model;

public:
	[[nodiscard]]
	GNeuro::Model<value_t> Model() const { return m_model; }

	/*
	 * Create a new model for the network.
	 */
	void CreateModel(const typename GNeuro::FunctionType<value_t>::loss_t _lossFunction, const GMath::size_t &_inputCount, const bool _randomize) {
		if (!_lossFunction) {
			throw NetworkError("No loss function provided.");
		}

		if (_inputCount < 1) {
			throw NetworkError("Invalid input count.");
		}

		m_model.SetLossFunction(_lossFunction);
		m_model.Fit(_inputCount);
		if (_randomize) m_model.Randomize();
	}

	/*
	 * Reset model to invalid state.
	 */
	void ResetModel() {
		m_model.Reset();
	}

	/*
	 * Calculate the batch mean loss for the model.
	 */
	[[nodiscard]]
	value_t BatchMeanLoss(const GMath::Matrix<value_t> &_inputBatches, const GMath::Matrix<value_t> &_expectedOutputBatches) {
		if (_inputBatches.Shape().Rows < 1) {
			throw NetworkError("No inputs given.");
		}

		if (_inputBatches.Shape().Columns != m_model.Inputs()) {
			throw NetworkError("Input batches sample size does not match model input amount.");
		}

		if (_inputBatches.Shape().Rows != _expectedOutputBatches.Shape().Rows) {
			throw NetworkError("Input batch count does not match output batch count.");
		}

		value_t meanLoss = 0;

		for (GMath::size_t i = 0; i < _inputBatches.Shape().Rows; i++) {
			meanLoss += m_model.MeanLoss(_inputBatches[i], _expectedOutputBatches[i]);
		}

		meanLoss /= _inputBatches.Shape().Rows;
		return meanLoss;
	}

	/*
	 * Add a new layer to the model.
	 */
	void AddLayer(const GMath::size_t &_neuronCount, const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {
		if (_neuronCount < 1) {
			throw NetworkError("Invalid neuron count given.");
		}

		if (!_activationFunction) {
			throw NetworkError("No activation function given.");
		}

		m_model.AddLayer(_neuronCount, _activationFunction);
	}

	/*
	 * Remove a layer from the model.
	 */
	void RemoveLayer(const GMath::size_t _index) {
		if (_index < 1 || _index >= m_model.Layers()) {
			throw NetworkError("Index out of bounds.");	
		}

		m_model.RemoveLayer(_index);
	}

	/*
	 * Calculate the output batches from inputs.
	 */
	[[nodiscard]]
	GMath::Matrix<value_t> Calculate(const GMath::Matrix<value_t> &_inputBatches) const {
		GMath::Matrix<value_t> output;

		if (_inputBatches.Shape().Rows < 1) {
			throw NetworkError("No input batches given.");
		}

		if (_inputBatches.Shape().Columns != m_model.Inputs()) {
			throw NetworkError("Input amount does not match model.");
		}

		for (GMath::size_t i = 0; i < _inputBatches.Shape().Rows; i++) {
			output = output.AppendRow(m_model.FeedForward(_inputBatches[i])[0]);
		}

		return output;
	}

	/*
	 * Train the model on input batches for [_epochCount] epochs.
	 */
	void Train(const GMath::Matrix<value_t> &_inputBatches, const GMath::Matrix<value_t> &_expectedOutputBatches, const value_t &_learningRate, const GMath::size_t &_epochCount) {
		if (_inputBatches.Shape().Rows != _expectedOutputBatches.Shape().Rows) {
			throw NetworkError("Input batch count does not match output batch count.");
		}

		if (_inputBatches.Shape().Columns != m_model.Inputs()) {
			throw NetworkError("Input amount does not match model.");
		}

		if (_inputBatches.Shape().Rows < 1) {
			throw NetworkError("No input batches provided.");
		}

		if (_epochCount < 1) {
			throw NetworkError("Epoch count is less than 1.");
		}

		if (_learningRate <= 0) {
			throw NetworkError("Learning rate has to be > 0.");
		}

		for (GMath::size_t i = 0; i < _epochCount; i++) {
			for (GMath::size_t j = 0; j < _inputBatches.Shape().Rows; j++) {
				m_model.BackPropagate(_inputBatches[j], _expectedOutputBatches[j], _learningRate);
			}

			std::cout << "Epoch: " << i << std::endl;
			std::cout << "\x1b[1F";
		}

		std::cout << std::endl;
	}

	/*
	 * Train the model on input batches until [_lossThreshold] batch mean loss is reached.
	 */
	void Train(const GMath::Matrix<value_t> &_inputBatches, const GMath::Matrix<value_t> &_expectedOutputBatches, const value_t &_learningRate, const value_t _lossThreshold, std::atomic<bool> &_running) {
		if (_inputBatches.Shape().Rows != _expectedOutputBatches.Shape().Rows) {
			throw NetworkError("Input batch count does not match output batch count.");
		}

		if (_inputBatches.Shape().Columns != m_model.Inputs()) {
			throw NetworkError("Input amount does not match model.");
		}

		if (_inputBatches.Shape().Rows < 1) {
			throw NetworkError("No input batches given.");
		}

		if (_learningRate <= 0) {
			throw NetworkError("Learning rate has to be > 0.");
		}

		if (_lossThreshold <= 0) {
			throw NetworkError("Loss threshold has to be > 0.");
		}

		value_t loss = 0;

		do {
			for (GMath::size_t j = 0; j < _inputBatches.Shape().Rows; j++) {
				m_model.BackPropagate(_inputBatches[j], _expectedOutputBatches[j], _learningRate);
			}

			loss = BatchMeanLoss(_inputBatches, _expectedOutputBatches);

			std::cout << "Mean Loss: " << loss << std::endl;
			std::cout << "\x1b[1F";
		} while (loss > _lossThreshold && _running);

		std::cout << std::endl;
	}
};
} // namespace GNeuro
