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
private:
	Model<value_t> m_model;

public:
	void CreateModel(const typename GNeuro::FunctionType<value_t>::loss_t _lossFunction, const GMath::size_t &_inputCount, const bool _randomize) {
		m_model.SetLossFunction(_lossFunction);
		m_model.Fit(_inputCount);
		if (_randomize) m_model.Randomize();
	}

	void ResetModel() {
		m_model.Reset();
	}

	value_t MeanLoss(const GMath::Matrix<value_t> &_inputBatches, const GMath::Matrix<value_t> &_expectedOutputBatches) {
		if (_inputBatches.Shape().Rows != _expectedOutputBatches.Shape().Rows) {
			throw std::runtime_error("Input batch count does not match output batch count.");
		}

		value_t meanLoss = 0;

		for (GMath::size_t i = 0; i < _inputBatches.Shape().Rows; i++) {
			meanLoss += m_model.MeanLoss(_inputBatches[i], _expectedOutputBatches[i]);
		}

		meanLoss /= _inputBatches.Shape().Rows;
		return meanLoss;
	}

	void AddLayer(const GMath::size_t &_neuronCount, const typename GNeuro::FunctionType<value_t>::activation_t _activationFunction) {
		m_model.AddLayer(_neuronCount, _activationFunction);
	}

	void RemoveLayer(const GMath::size_t _index) {
		m_model.RemoveLayer(_index);
	}

	GMath::Matrix<value_t> Calculate(const GMath::Matrix<value_t> &_inputBatches) {
		GMath::Matrix<value_t> output;

		for (GMath::size_t i = 0; i < _inputBatches.Shape().Rows; i++) {
			output = output.AppendRow(m_model.FeedForward(_inputBatches[i])[0]);
		}

		return output;
	}

	void Train(const GMath::Matrix<value_t> &_inputBatches, const GMath::Matrix<value_t> &_expectedOutputBatches, const value_t &_learningRate, const GMath::size_t &_epochCount) {
		if (_inputBatches.Shape().Rows != _expectedOutputBatches.Shape().Rows) {
			throw std::runtime_error("Input batch count does not match output batch count.");
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

	void Train(const GMath::Matrix<value_t> &_inputBatches, const GMath::Matrix<value_t> &_expectedOutputBatches, const value_t &_learningRate, const value_t _lossThreshold, std::atomic<bool> &_running) {
		if (_inputBatches.Shape().Rows != _expectedOutputBatches.Shape().Rows) {
			throw std::runtime_error("Input batch count does not match output batch count.");
		}

		value_t loss = 0;

		do {
			for (GMath::size_t j = 0; j < _inputBatches.Shape().Rows; j++) {
				m_model.BackPropagate(_inputBatches[j], _expectedOutputBatches[j], _learningRate);
			}

			loss = MeanLoss(_inputBatches, _expectedOutputBatches);

			std::cout << "Mean Loss: " << loss << std::endl;
			std::cout << "\x1b[1F";
		} while (loss > _lossThreshold && _running);

		std::cout << std::endl;
	}
};
} // namespace GNeuro
