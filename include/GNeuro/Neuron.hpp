#pragma once
#include "Type.hpp"
#include <vector>

namespace GNeuro {
  class Neuron {
  public:
    Neuron() : m_bias(0), m_weights(0), m_activation(nullptr) {};
    Neuron(Neuron &&_n) = default;
    Neuron(const Neuron &_n) = default;
    Neuron &operator=(Neuron &&_n) = default;
    Neuron &operator=(const Neuron &_n) = default;
    ~Neuron() = default;

    const size_t GetInputsCount() const { return m_weights.size(); };
    const DECIMAL_T GetBias() const { return m_bias; }
    const DECIMAL_T GetWeight(const size_t _inputIndex) const { return m_weights[_inputIndex]; }
    const ACTIVATION_T GetActivation() const { return m_activation; }

    void SetInputsCount(const size_t _count) { m_weights.resize(_count); };
    void SetBias(const DECIMAL_T _value) { m_bias = _value; }
    void SetWeight(const size_t _inputIndex, const DECIMAL_T _value) { m_weights[_inputIndex] = _value; }
    void SetActivation(const ACTIVATION_T _activation) { m_activation = _activation; }

  private:
    std::vector<DECIMAL_T> m_weights;
    DECIMAL_T m_bias;
    ACTIVATION_T m_activation;
  };
}
