#pragma once
#include "GNeuro/Neuron.hpp"
#include <vector>

namespace GNeuro {
template <typename value_t> class Layer {
private:
  typedef value_t (*activation_t)(value_t _in, bool _derived, std::string &_funcName);

public:
  Layer() = default;
  explicit Layer(const size_t _neuronCount) : m_neurons(_neuronCount) {};
  Layer(Layer &&) = default;
  Layer(const Layer &) = default;
  Layer &operator=(Layer &&) = default;
  Layer &operator=(const Layer &) = default;
  ~Layer() = default;

  const size_t GetSize() const { return m_neurons.size(); };

  void SetActivationFunction(const activation_t _func) {
    for (size_t __neuronIndex = 0; __neuronIndex < GetSize(); __neuronIndex++) {
      auto &neuron = operator[](__neuronIndex);

      neuron.SetActivation(_func);
    }
  }

  Neuron<value_t> &operator[](const size_t _neuron) {
    return m_neurons[_neuron];
  }
  const Neuron<value_t> &operator[](const size_t _neuron) const {
    return m_neurons[_neuron];
  }

private:
  std::vector<Neuron<value_t>> m_neurons;
};
} // namespace GNeuro
