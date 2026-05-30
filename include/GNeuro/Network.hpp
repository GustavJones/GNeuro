/*
 * This file includes the GNeuro::Network class.
 */

#pragma once
#include "GMath/Matrix.hpp"
#include "GMath/Types.hpp"
#include "GNeuro/Functions.hpp"
#include "GNeuro/Model.hpp"
#include <atomic>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace GNeuro {

static std::atomic<bool> s_training = false;

static void TrainingSignalHandler(int _) { s_training = false; }

/*
 * An object to create a Neural Network. Uses different GNeuro::Layers to define
 * a network that can be created, trained, used for calculations, cleared, saved
 * and loaded.
 */
template <typename value_t> class Network {
private:
  GNeuro::Model<value_t> m_model;
  typename Functions<value_t>::loss_t m_loss = nullptr;

public:
  Network() = default;
  Network(Network &&) = default;
  Network(const Network &) = default;
  Network &operator=(Network &&) = default;
  Network &operator=(const Network &) = default;
  ~Network() = default;

  /*
   * Set the loss function used for training and loss calculation.
   */
  void SetLoss(const typename Functions<value_t>::loss_t _loss) {
    m_loss = _loss;
  }

  /*
   * Get the loss function used for training and loss calculation.
   */
  const typename Functions<value_t>::loss_t GetLoss() const { return m_loss; }

  /*
   * Calculate the outputs from an input batch.
   */
  [[nodiscard]]
  GMath::DynamicArray<value_t> Calculate(const GMath::Matrix<value_t> &_inputs) {
    GMath::DynamicArray<GMath::DynamicArray<value_t>> output = m_model.Calculate(_inputs);

    if (output.Size() <= 0) {
      throw std::runtime_error("Unknown error when calculating outputs from network...");
    }

    return output[output.Size() - 1];
  };

  /*
   * Continuously train the network on a collection of input and expected output
   * values until a certain loss threshold is reached.
   * Use _learningRate to adjust the model weights and biases.
   * Use _learningRateChangeFactor for variable learning rate change. (set to <=
   * 0 to disable)
   */
  void Train(const GMath::Matrix<value_t> &_inputsBatch,
             const GMath::Matrix<value_t> &_expectedOutputsBatch,
             double _learningRate, const double _lossThreshold) {
    if (_inputsBatch.Shape().Rows != _expectedOutputsBatch.Shape().Rows) {
      throw std::runtime_error(
          "Inputs and expected outputs batch size doesn't match.");
    }

    while (s_training) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    s_training = true;
    std::signal(SIGINT, TrainingSignalHandler);
    try {
      value_t meanLoss = _lossThreshold + 1;
      value_t previousMeanLoss = MeanLoss(_inputsBatch, _expectedOutputsBatch);
      std::cout << "Original Loss: " << GParsing::to_string(previousMeanLoss) << '\t' << "Original Learning Rate: " << GParsing::to_string(_learningRate) << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;

      while (meanLoss > _lossThreshold && s_training) {
        for (size_t __batchIndex = 0; __batchIndex < _inputsBatch.Shape().Rows; __batchIndex++) {
          m_model = BackPropagate(_inputsBatch[__batchIndex], _expectedOutputsBatch[__batchIndex], _learningRate);
        }

        meanLoss = MeanLoss(_inputsBatch, _expectedOutputsBatch);

        std::cout << "\x1b[2F";
        std::cout << "Loss: " << GParsing::to_string(meanLoss) << '\n';
        std::cout << "Learning Rate: " << GParsing::to_string(_learningRate) << std::endl;
        previousMeanLoss = meanLoss;
      }
    } catch (...) {
      s_training = false;
      std::signal(SIGINT, SIG_DFL);
      throw;
      return;
    }

    s_training = false;
    std::signal(SIGINT, SIG_DFL);
  }

  /*
   * Mutates the network then passes the network to the _callback function for
   * evaluation. The _callback function should return a coefficient that is used
   * to save the mutated network weights and biases as a weighted amount of the
   * _callback output. e.g. if the _callback -> 0.1 then the network will be
   * saved with 0.1 * mutate amount.
   */
  // TODO: Implementation
  [[deprecated]]
  void Mutate(const typename Functions<value_t>::mutate_t _callback) {
    // TODO move mutate to model class
    // const value_t mutateAmount = Random(-1.0, 1.0);
    //
    // const size_t layerIndex = std::round(Random(0, GetLayersCount() - 1));
    // auto &layer = operator[](layerIndex);
    // const size_t neuronIndex = std::round(Random(0, layer.GetSize() - 1));
    // auto &neuron = layer[neuronIndex];
    // const size_t attributeIndex =
    //     std::round(Random(0, neuron.GetInputsCount()));
    //
    // if (attributeIndex <= 0) {
    //   auto originalValue = neuron.GetBias();
    //   neuron.SetBias(originalValue + mutateAmount);
    //
    //   auto reward = _callback(*this);
    //
    //   neuron.SetBias(originalValue + (mutateAmount * reward));
    // } else {
    //   const size_t weightIndex = attributeIndex - 1;
    //
    //   auto originalValue = neuron.GetWeight(weightIndex);
    //   neuron.SetWeight(weightIndex, originalValue + mutateAmount);
    //
    //   auto reward = _callback(*this);
    //   neuron.SetWeight(weightIndex, originalValue + (mutateAmount * reward));
    //
    //   neuron.SetWeight(weightIndex, originalValue + (mutateAmount * reward));
    // }
  }

  /*
   * Calculates the average loss for each input / expected output pair.
   */
  [[nodiscard]]
  value_t MeanLoss(const GMath::Matrix<value_t> &_inputsBatch, const GMath::Matrix<value_t> &_expectedOutputsBatch) {
    if (!m_loss) {
      throw std::runtime_error("No loss function provided.");
    }

    return m_model.MeanLoss(_inputsBatch, _expectedOutputsBatch, m_loss);
  }

  /*
   * Clear the network model.
   */
  void ClearModel() { m_model = GNeuro::Model<value_t>(); }

  /*
   * Set the model of the network.
   */
  void SetModel(const GNeuro::Model<value_t> &_m) { m_model = _m; }

  /*
   * Get the model of the network.
   */
  [[nodiscard]]
  GNeuro::Model<value_t> GetModel() const {
    return m_model;
  }

private:
  [[nodiscard]]
  GNeuro::Model<value_t> BackPropagate(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expectedOutputs, const double _learningRate) {
    if (!m_loss) {
      throw std::runtime_error("No loss function set to network...");
    }

    if (!_inputs.IsColumnMatrix() && !_inputs.IsRowMatrix()) {
      throw std::runtime_error("More than one input batch given...");
    }

    if (_inputs.IsColumnMatrix() && !_inputs.IsRowMatrix()) {
      throw std::runtime_error("Inputs is a column matrix. Not yet implemented...");
    }

    if (!_expectedOutputs.IsColumnMatrix() && !_expectedOutputs.IsRowMatrix()) {
      throw std::runtime_error("More than one expected output batch given...");
    }

    if (_expectedOutputs.IsColumnMatrix() && !_expectedOutputs.IsRowMatrix()) {
      throw std::runtime_error("Expected outputs is a column matrix. Not yet implemented...");
    }

    GMath::DynamicArray<GMath::DynamicArray<value_t>> uOutputs;
    GMath::DynamicArray<GMath::DynamicArray<value_t>> aOutputs;
    m_model.Calculate(_inputs, uOutputs, aOutputs);

    GMath::DynamicArray<GMath::DynamicArray<value_t>> gradients = aOutputs;

    // Calculate the gradients from the last layer
    const GMath::size_t lastLayerIndex = m_model.GetLayerCount() - 1;
    std::string _;

    for (size_t __neuronIndex = 0; __neuronIndex < m_model.GetNeuronCount(lastLayerIndex); __neuronIndex++) {
      auto activationFunction = m_model.GetActivationFunction(lastLayerIndex, __neuronIndex);

      const auto lossDelta = m_loss(aOutputs[lastLayerIndex][__neuronIndex], _expectedOutputs[__neuronIndex][0], true, _);

      if (activationFunction) {
        const auto activationDelta = activationFunction(uOutputs[lastLayerIndex][__neuronIndex], true, _);
        gradients[lastLayerIndex][__neuronIndex] = lossDelta * activationDelta;
      } else {
        gradients[lastLayerIndex][__neuronIndex] = lossDelta;
      }
    }

    // Now repeat for every layer
    for (int64_t __layerIndex = m_model.GetLayerCount() - 2; __layerIndex >= 0; __layerIndex--) {
      for (size_t __neuronIndex = 0; __neuronIndex < m_model.GetNeuronCount(__layerIndex); __neuronIndex++) {
        auto activationFunction = m_model.GetActivationFunction(__layerIndex, __neuronIndex);
        value_t parentDelta = 0;

        for (size_t __parentNeuronIndex = 0; __parentNeuronIndex < m_model.GetNeuronCount(__layerIndex + 1); __parentNeuronIndex++) {
          parentDelta += gradients[__layerIndex + 1][__parentNeuronIndex] * m_model.GetWeight(__layerIndex + 1, __parentNeuronIndex, __neuronIndex);
        }

        if (activationFunction) {
          const auto activationDelta = activationFunction(uOutputs[__layerIndex][__neuronIndex], true, _);
          gradients[__layerIndex][__neuronIndex] = parentDelta * activationDelta;
        } else {
          gradients[__layerIndex][__neuronIndex] = parentDelta;
        }
      }
    }

    GNeuro::Model<value_t> output = m_model;

    // Actually update the output model
    for (int64_t __layerIndex = output.GetLayerCount() - 1; __layerIndex >= 0; __layerIndex--) {
      for (size_t __neuronIndex = 0; __neuronIndex < output.GetNeuronCount(__layerIndex); __neuronIndex++) {

        const auto newBias = output.GetBias(__layerIndex, __neuronIndex) - (gradients[__layerIndex][__neuronIndex] * _learningRate);
        output.SetBias(newBias, __layerIndex, __neuronIndex);

        for (size_t __weightIndex = 0; __weightIndex < output.GetWeightCount(__layerIndex); __weightIndex++) {
          value_t newWeight;
          if (__layerIndex > 0) {
            newWeight = output.GetWeight(__layerIndex, __neuronIndex, __weightIndex) - (gradients[__layerIndex][__neuronIndex] * aOutputs[__layerIndex - 1][__weightIndex] * _learningRate);
          } else {
            // TODO _inputs[0] should be changed to work with column matrices as well
            newWeight = output.GetWeight(__layerIndex, __neuronIndex, __weightIndex) - (gradients[__layerIndex][__neuronIndex] * _inputs[0][__weightIndex] * _learningRate);
          }

          output.SetWeight(newWeight, __layerIndex, __neuronIndex, __weightIndex);
        }
      }
    }

    return output;
  }
};
} // namespace GNeuro
