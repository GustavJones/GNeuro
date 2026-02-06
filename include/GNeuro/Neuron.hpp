#pragma once
#include <vector>
#include <string>

namespace GNeuro {
  template<typename value_t>
  class Neuron {
  private:
    typedef value_t (*activation_t)(value_t _in, bool _derived, std::string &_funcName);
  public:
    Neuron() : m_bias(0), m_weights(0), m_activation(nullptr) {};
    Neuron(Neuron &&_n) = default;
    Neuron(const Neuron &_n) = default;
    Neuron &operator=(Neuron &&_n) = default;
    Neuron &operator=(const Neuron &_n) = default;
    ~Neuron() = default;

    const size_t GetInputsCount() const { return m_weights.size(); };
    const value_t GetBias() const { return m_bias; }
    const value_t GetWeight(const size_t _inputIndex) const { return m_weights[_inputIndex]; }
    const activation_t GetActivation() const { return m_activation; }

    void SetInputsCount(const size_t _count) { m_weights.resize(_count); };
    void SetBias(const value_t _value) { m_bias = _value; }
    void SetWeight(const size_t _inputIndex, const value_t _value) { m_weights[_inputIndex] = _value; }
    void SetActivation(const activation_t _activation) { m_activation = _activation; }

  private:
    std::vector<value_t> m_weights;
    value_t m_bias;
    activation_t m_activation;
  };
}
