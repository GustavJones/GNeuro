#pragma once

#include "GNeuro/Type.hpp"
#include "Neuron.hpp"
#include <vector>

namespace GNeuro {
class Layer {
public:
  Layer() = default;
  explicit Layer(const size_t _neuronCount) : m_neurons(_neuronCount) {};
  Layer(Layer &&) = default;
  Layer(const Layer &) = default;
  Layer &operator=(Layer &&) = default;
  Layer &operator=(const Layer &) = default;
  ~Layer() = default;

  const size_t GetSize() const { return m_neurons.size(); };

  void SetActivationFunction(const ACTIVATION_T _func);

  Neuron &operator[](const size_t _neuron) { return m_neurons[_neuron]; }
  const Neuron &operator[](const size_t _neuron) const {
    return m_neurons[_neuron];
  }

private:
  std::vector<Neuron> m_neurons;
};
} // namespace GNeuro
