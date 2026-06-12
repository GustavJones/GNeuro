/*
 * This file includes the GNeuro::Layer class.
 */

#pragma once
#include "GMath/Matrix.hpp"
#include "GMath/Types.hpp"
#include "GNeuro/Random.hpp"
#include "GNeuro/Functions.hpp"
#include <cstddef>
#include <stdexcept>

namespace GNeuro {
/*
 * A single layer in the GNeuro::Network class. Contains the GNeuro::Neuron
 * objects contained in the layer of the network.
 */
template <typename value_t> class Layer {
public:

private:
  GMath::DynamicArray<typename Functions<value_t>::activation_t> m_activationFunctions;
  GMath::Matrix<value_t> m_weights;
  GMath::Matrix<value_t> m_biases;

public:
  Layer() = default;

  explicit Layer(const GMath::size_t _neuronCount,
                 const typename Functions<value_t>::activation_t _activationFunction = nullptr)
      : m_weights(_neuronCount, 0), m_biases(_neuronCount, 1),
        m_activationFunctions(_neuronCount) {
    for (size_t i = 0; i < _neuronCount; i++) {
      SetActivationFunction(_activationFunction, i);
    }
  };

  Layer(Layer &&) = default;
  Layer(const Layer &) = default;
  Layer &operator=(Layer &&) = default;
  Layer &operator=(const Layer &) = default;
  ~Layer() = default;

  /*
   * Get the amount of neurons in this layer.
   */
  [[nodiscard]]
  GMath::size_t GetSize() const { return m_biases.Shape().Rows; }

  /*
   * Set the amount of neuron in this layer.
   */
  void SetSize(const GMath::size_t _count) {
    m_weights.Reshape({_count, 0});
    m_biases.Reshape({_count, 1});
    m_activationFunctions.Resize(_count);
  }

  /*
   * Get the amount of weights for each neuron in the layer.
   */
  [[nodiscard]]
  GMath::size_t GetWeightCount() const {
    return m_weights.Shape().Columns;
  }

  /*
   * Get the activation function for a specific neuron in the layer.
   */
  [[nodiscard]]
  typename Functions<value_t>::activation_t GetActivationFunction(const GMath::size_t _index) const { return m_activationFunctions[_index]; }

  /*
   * Sets the activation function of the neurons in the layer
   * individually.
   */
  void SetActivationFunction(const typename Functions<value_t>::activation_t _func, const GMath::size_t _index) {
    m_activationFunctions[_index] = _func;
  }

  /*
   * Get the weight for a specific neuron and weight index.
   */
  [[nodiscard]]
  value_t GetWeight(const GMath::size_t _neuron, const GMath::size_t _weight) const {
    return m_weights[_neuron][_weight];
  }

  /*
   * Set the weight for a specific neuron and weight index.
   */
  void SetWeight(const value_t _value, const GMath::size_t _neuron, const GMath::size_t _weight) {
    m_weights[_neuron][_weight] = _value;
  }

  /*
   * Get the bias for a specific neuron index.
   */
  [[nodiscard]]
  value_t GetBias(const GMath::size_t _neuron) const {
    return m_biases[_neuron][0];
  }

  /*
   * Set the bias for a specific neuron index.
   */
  void SetBias(const value_t _value, const GMath::size_t _neuron) {
    m_biases[_neuron] = _value;
  }

  /*
   * Adjust weight counts for the amount of inputs the layer will recieve.
   */
  void FitLayer(const GMath::size_t _inputCount) {
    m_weights.Reshape({m_weights.Shape().Rows, _inputCount});
  }

  /*
   * Randomize the weights and biases of neurons.
   */
  void Randomize() {
    auto neuronCount = m_biases.Shape().Rows;
    auto neuronWeightCount = m_weights.Shape().Columns;

    for (size_t i = 0; i < neuronCount; i++) {
      m_biases[i][0] = Random(0.0, 1.0);
      for (size_t j = 0; j < neuronWeightCount; j++) {
        m_weights[i][j] = Random(0.0, 1.0);
      }
    }
  }

  /*
   * Run input values through the layer.
   * Output is given as a row matrix.
   */
  void Calculate(const GMath::Matrix<value_t> &_inputs, GMath::Matrix<value_t> &_unactivated, GMath::Matrix<value_t> &_activated) const {
    std::string _;

    if (_inputs.IsRowMatrix()) {
      _unactivated = _inputs * m_weights.Transpose() + m_biases.Transpose();
    } else if (_inputs.IsColumnMatrix()) {
      _unactivated = _inputs.Transpose() * m_weights.Transpose() + m_biases.Transpose();
    }
    else {
      throw std::runtime_error("Inputs not a column / row matrix.");
    }

    _activated.Reshape(_unactivated.Shape());

    for (size_t i = 0; i < _unactivated.Shape().Columns; i++) {
      typename Functions<value_t>::activation_t func = GetActivationFunction(i);
      value_t val = _unactivated[0][i];

      if (func) {
        _activated[0][i] = func(val, false, _);
      }
    }
  }

  /*
   * Run input values through the layer.
   * Output is given as a row matrix.
   */
  [[nodiscard]]
  GMath::Matrix<value_t> Calculate(const GMath::Matrix<value_t> &_inputs) const {
    GMath::Matrix<value_t> uOutputs, aOutputs;
    Calculate(_inputs, uOutputs, aOutputs);
    return aOutputs;
  }
};
} // namespace GNeuro
