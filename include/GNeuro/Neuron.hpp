/*
 * This file includes the GNeuro::Neuron class.
 */

#pragma once
#include <vector>
#include <string>

namespace GNeuro {
  /*
   * A single neuron in the GNeuro::Layer class. Contains the weights, bias and
   * activation function of the neuron.
   */
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

    /*
     * Get the amount of inputs this neuron supports. (the amount of weights in the neuron)
     */
    const size_t GetInputsCount() const { return m_weights.size(); };

    /*
     * Get the bias value of the neuron.
     */
    const value_t GetBias() const { return m_bias; }

    /*
     * Get the weight value of a specific input index of the neuron.
     */
    const value_t GetWeight(const size_t _inputIndex) const { return m_weights[_inputIndex]; }

    /*
     * Get the activation function used by the neuron.
     */
    const activation_t GetActivation() const { return m_activation; }

    /*
     * Set the amount of inputs this neuron supports. (the amount of weights in the neuron)
     */
    void SetInputsCount(const size_t _count) { m_weights.resize(_count); };

    /*
     * Set the bias value of the neuron.
     */
    void SetBias(const value_t _value) { m_bias = _value; }

    /*
     * Set the weight value of a specific input index of the neuron.
     */
    void SetWeight(const size_t _inputIndex, const value_t _value) { m_weights[_inputIndex] = _value; }

    /*
     * Set the activation function of the neuron.
     */
    void SetActivation(const activation_t _activation) { m_activation = _activation; }
  private:
    std::vector<value_t> m_weights;
    value_t m_bias;
    activation_t m_activation;
  };
}
