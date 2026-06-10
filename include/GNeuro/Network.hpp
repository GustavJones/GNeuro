/*
 * This file includes the GNeuro::Network class.
 */

#pragma once
#include "GMath/DynamicArray.hpp"
#include "GMath/Matrix.hpp"
#include "GMath/Types.hpp"
#include "GNeuro/Functions.hpp"
#include "GNeuro/Model.hpp"
#include "GNeuro/Random.hpp"
#include <algorithm>
#include <cstdint>
#include <future>
#include <mutex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace GNeuro {

/*
 * An object to create a Neural Network. Uses different GNeuro::Layers to define
 * a network that can be created, trained, used for calculations, cleared, saved
 * and loaded.
 */
template <typename value_t> class Network {
private:
  mutable std::mutex m_training;
  mutable std::mutex m_mutex;
  GNeuro::Model<value_t> m_model;
  typename Functions<value_t>::loss_t m_loss = nullptr;

public:
  Network() = default;

  Network(Network &&_n) noexcept {
    std::lock_guard<std::mutex> guard(_n.m_mutex);

    m_model = std::move(_n.m_model);
    m_loss = _n.m_loss;
    _n.m_loss = nullptr;
  };

  Network(const Network &_n) {
    std::lock_guard<std::mutex> guard(_n.m_mutex);

    m_model = _n.m_model;
    m_loss = _n.m_loss;
  };

  Network &operator=(Network &&_n) noexcept {
    if (this != &_n) {
      GNeuro::Model<value_t> m;
      typename Functions<value_t>::loss_t l;

      {
        std::lock_guard<std::mutex> guard(_n.m_mutex);
        m = std::move(_n.m_model);
        l = _n.m_loss;
        _n.m_loss = nullptr;
      }

      {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_model = std::move(m);
        m_loss = l;
      }
    }

    return *this;
  };

  Network &operator=(const Network &_n) {
    if (this != &_n) {
      GNeuro::Model<value_t> m;
      typename Functions<value_t>::loss_t l;

      {
        std::lock_guard<std::mutex> guard(_n.m_mutex);
        m = _n.m_model;
        l = _n.m_loss;
      }

      {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_model = std::move(m);
        m_loss = l;
      }
    }

    return *this;
  };

  ~Network() = default;

  /*
   * Set the loss function used for training and loss calculation.
   */
  void SetLoss(const typename Functions<value_t>::loss_t _loss) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_loss = _loss;
  }

  /*
   * Get the loss function used for training and loss calculation.
   */
  typename Functions<value_t>::loss_t GetLoss() const { 
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_loss; 
  }

  /*
   * Calculate the outputs from an input batch.
   */
  [[nodiscard]]
  GMath::DynamicArray<value_t> Calculate(const GMath::Matrix<value_t> &_inputs) const {

    GNeuro::Model<value_t> copy;

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      copy = m_model;
    }

    GMath::DynamicArray<GMath::DynamicArray<value_t>> output = copy.Calculate(_inputs);

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
   * Can NOT be called in async. (Will be blocked by static variable check)
   */
  void Train(const GMath::Matrix<value_t> &_inputsBatch,
             const GMath::Matrix<value_t> &_expectedOutputsBatch,
             double _learningRate, const double _lossThreshold, const bool _randomSubBatching = true) {
    std::lock_guard<std::mutex> guard(m_training);
    const auto BATCH_COUNT = _inputsBatch.Shape().Rows;

    if (BATCH_COUNT != _expectedOutputsBatch.Shape().Rows) {
      throw std::runtime_error("Inputs and expected outputs batch size doesn't match.");
    }

    if (BATCH_COUNT == 0) {
      throw std::runtime_error("No training batches provided...");
    }

    value_t meanLoss = _lossThreshold + 1;
    value_t previousMeanLoss = MeanLoss(_inputsBatch, _expectedOutputsBatch);
    std::cout << "Original Loss: " << GParsing::to_string(previousMeanLoss) << '\t' << "Original Learning Rate: " << GParsing::to_string(_learningRate) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    while (meanLoss > _lossThreshold) {
      GMath::DynamicArray<std::future<GNeuro::Model<value_t>>> threads;
      GMath::DynamicArray<GNeuro::Model<value_t>> models;

      GMath::size_t batchStart = GNeuro::Random(0.0, (double)BATCH_COUNT - 1);
      GMath::size_t batchEnd = GNeuro::Random((double)batchStart, (double)BATCH_COUNT - 1);

      if (_randomSubBatching) {
        // Random sub-batch run
        for (size_t __batchIndex = batchStart; __batchIndex < batchEnd + 1; __batchIndex++) {
          threads.EmplaceBack(std::async(std::launch::async, &Network::BackPropagate, this, (_inputsBatch[__batchIndex]), (_expectedOutputsBatch[__batchIndex]), _learningRate));
        }
      }

      // Full epoch 
      for (size_t __batchIndex = 0; __batchIndex < BATCH_COUNT; __batchIndex++) {
        threads.EmplaceBack(std::async(std::launch::async, &Network::BackPropagate, this, (_inputsBatch[__batchIndex]), (_expectedOutputsBatch[__batchIndex]), _learningRate));
      }

      for (size_t __threadIndex = 0; __threadIndex < threads.Size(); __threadIndex++) {
        models.PushBack(threads[__threadIndex].get());
      }

      {
        std::lock_guard<std::mutex> guard(m_mutex);

        m_model = Average(models);
      }

      meanLoss = MeanLoss(_inputsBatch, _expectedOutputsBatch);

      std::cout << "\x1b[2F";
      std::cout << "Loss: " << GParsing::to_string(meanLoss) << '\n';
      std::cout << "Learning Rate: " << GParsing::to_string(_learningRate) << std::endl;
      previousMeanLoss = meanLoss;
    }
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
  value_t MeanLoss(const GMath::Matrix<value_t> &_inputsBatch, const GMath::Matrix<value_t> &_expectedOutputsBatch) const {
    GNeuro::Model<value_t> copy;
    typename Functions<value_t>::loss_t lossFunc;

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      if (!m_loss) {
        throw std::runtime_error("No loss function provided.");
      }

      copy = m_model;
      lossFunc = m_loss;
    }

    return copy.MeanLoss(_inputsBatch, _expectedOutputsBatch, lossFunc);
  }

  /*
   * Clear the network model.
   */
  void ClearModel() { 
    std::lock_guard<std::mutex> guard(m_mutex);
    m_model = GNeuro::Model<value_t>(); 
  }

  /*
   * Set the model of the network.
   */
  void SetModel(const GNeuro::Model<value_t> &_m) { 
    std::lock_guard<std::mutex> guard(m_mutex);
    m_model = _m; 
  }

  /*
   * Get the model of the network.
   */
  [[nodiscard]]
  GNeuro::Model<value_t> GetModel() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_model;
  }

private:
  /*
   * Calculate the average parameter values between all the models.
   * Only change weights and biases. The rest of the data will contain the first model's values (e.g. activations).
   */
  [[nodiscard]]
  GNeuro::Model<value_t> Average(const GMath::DynamicArray<GNeuro::Model<value_t>> &_models) const {
    if (_models.Size() < 1) {
      throw std::runtime_error("No models to calculate average...");
    }

    // Copy the first model to obtain the correct shape and activations.
    std::mutex avg_mutex;
    GNeuro::Model<value_t> avg = _models[0];

    // Devide first model with model count.
    for (size_t __layerIndex = 0; __layerIndex < avg.GetLayerCount(); __layerIndex++) {
      for (size_t __neuronIndex = 0; __neuronIndex < avg.GetNeuronCount(__layerIndex); __neuronIndex++) {
        value_t bias = avg.GetBias(__layerIndex, __neuronIndex);
        bias /= _models.Size();
        avg.SetBias(bias, __layerIndex, __neuronIndex);

        for (size_t __weightIndex = 0; __weightIndex < avg.GetWeightCount(__layerIndex); __weightIndex++) {
          value_t weight = avg.GetWeight(__layerIndex, __neuronIndex, __weightIndex);
          weight /= _models.Size();
          avg.SetWeight(weight, __layerIndex, __neuronIndex, __weightIndex);
        }
      }
    }

    // Add all of the other models to the average.
    GMath::DynamicArray<std::future<void>> threads(_models.Size() - 1);
    const auto threadFunc = [](const GMath::DynamicArray<GNeuro::Model<value_t>> *_models, const size_t __modelIndex, GNeuro::Model<value_t> *_avg, std::mutex *_mutex) {
      for (size_t __layerIndex = 0; __layerIndex < (*_models)[__modelIndex].GetLayerCount(); __layerIndex++) {
        for (size_t __neuronIndex = 0; __neuronIndex < (*_models)[__modelIndex].GetNeuronCount(__layerIndex); __neuronIndex++) {
          GMath::DynamicArray<value_t> weights((*_models)[__modelIndex].GetWeightCount(__layerIndex));
          value_t bias = (*_models)[__modelIndex].GetBias(__layerIndex, __neuronIndex) / _models->Size();

          for (size_t __weightIndex = 0; __weightIndex < (*_models)[__modelIndex].GetWeightCount(__layerIndex); __weightIndex++) {
            weights[__weightIndex] = (*_models)[__modelIndex].GetWeight(__layerIndex, __neuronIndex, __weightIndex) / _models->Size();
          }

          std::lock_guard<std::mutex> guard(*_mutex);
          _avg->SetBias(_avg->GetBias(__layerIndex, __neuronIndex) + bias, __layerIndex, __neuronIndex);

          for (size_t __weightIndex = 0; __weightIndex < weights.Size(); __weightIndex++) {
            _avg->SetWeight(_avg->GetWeight(__layerIndex, __neuronIndex, __weightIndex) + weights[__weightIndex], __layerIndex, __neuronIndex, __weightIndex);
          }
        }
      }
    };

    for (size_t __modelIndex = 1; __modelIndex < _models.Size(); __modelIndex++) {
      threads[__modelIndex - 1] = std::async(std::launch::async, threadFunc, &_models, __modelIndex, &avg, &avg_mutex);
    }

    for (size_t __threadIndex = 0; __threadIndex < threads.Size(); __threadIndex++) {
      threads[__threadIndex].wait();
    }

    return avg;
  }

  /*
   * Calculate a new model through back propagation to minimize the loss.
   * Can be called in async functions. 
   */
  [[nodiscard]]
  GNeuro::Model<value_t> BackPropagate(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expectedOutputs, const double _learningRate) const {
    typename Functions<value_t>::loss_t lossFunc;
    GNeuro::Model<value_t> output;

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      lossFunc = m_loss;
      output = m_model;
    }

    if (!lossFunc) {
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
    output.Calculate(_inputs, uOutputs, aOutputs);

    GMath::DynamicArray<GMath::DynamicArray<value_t>> gradients = aOutputs;

    // Calculate the gradients from the last layer
    const GMath::size_t lastLayerIndex = output.GetLayerCount() - 1;
    std::string _;

    for (size_t __neuronIndex = 0; __neuronIndex < output.GetNeuronCount(lastLayerIndex); __neuronIndex++) {
      auto activationFunction = output.GetActivationFunction(lastLayerIndex, __neuronIndex);

      const auto lossDelta = lossFunc(aOutputs[lastLayerIndex][__neuronIndex], _expectedOutputs[__neuronIndex][0], true, _);

      if (activationFunction) {
        const auto activationDelta = activationFunction(uOutputs[lastLayerIndex][__neuronIndex], true, _);
        gradients[lastLayerIndex][__neuronIndex] = lossDelta * activationDelta;
      } else {
        gradients[lastLayerIndex][__neuronIndex] = lossDelta;
      }
    }

    // Now repeat for every layer
    for (int64_t __layerIndex = output.GetLayerCount() - 2; __layerIndex >= 0; __layerIndex--) {
      for (size_t __neuronIndex = 0; __neuronIndex < output.GetNeuronCount(__layerIndex); __neuronIndex++) {
        auto activationFunction = output.GetActivationFunction(__layerIndex, __neuronIndex);
        value_t parentDelta = 0;

        for (size_t __parentNeuronIndex = 0; __parentNeuronIndex < output.GetNeuronCount(__layerIndex + 1); __parentNeuronIndex++) {
          parentDelta += gradients[__layerIndex + 1][__parentNeuronIndex] * output.GetWeight(__layerIndex + 1, __parentNeuronIndex, __neuronIndex);
        }

        if (activationFunction) {
          const auto activationDelta = activationFunction(uOutputs[__layerIndex][__neuronIndex], true, _);
          gradients[__layerIndex][__neuronIndex] = parentDelta * activationDelta;
        } else {
          gradients[__layerIndex][__neuronIndex] = parentDelta;
        }
      }
    }

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
