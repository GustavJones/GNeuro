/*
 * This file includes the GNeuro::Layer class.
 */

#pragma once
#include "GNeuro/Neuron.hpp"
#include <vector>

namespace GNeuro {
/*
 * A single layer in the GNeuro::Network class. Contains the GNeuro::Neuron
 * objects contained in the layer of the network.
 */
template <typename value_t> class Layer {
private:
  typedef value_t (*activation_t)(value_t _in, bool _derived, std::string &_funcName);
  std::vector<Neuron<value_t>> m_neurons;

public:
  Layer() = default;
  explicit Layer(const size_t _neuronCount) : m_neurons(_neuronCount) {};
  Layer(Layer &&) = default;
  Layer(const Layer &) = default;
  Layer &operator=(Layer &&) = default;
  Layer &operator=(const Layer &) = default;
  ~Layer() = default;

  /*
   * Get the amount of neurons in this layer.
   */
  const size_t GetSize() const { return m_neurons.size(); };

  /*
   * Sets the activation function for all of the neurons in the layer
   * individually.
   */
  void SetActivationFunction(const activation_t _func) {
    for (size_t __neuronIndex = 0; __neuronIndex < GetSize(); __neuronIndex++) {
      auto &neuron = operator[](__neuronIndex);

      neuron.SetActivation(_func);
    }
  }

  /*
   * Get the neuron at a specific index of the layer.
   */
  Neuron<value_t> &operator[](const size_t _neuron) {
    return m_neurons[_neuron];
  }

  /*
   * Get the neuron at a specific index of the layer.
   */
  const Neuron<value_t> &operator[](const size_t _neuron) const {
    return m_neurons[_neuron];
  }
};
} // namespace GNeuro
