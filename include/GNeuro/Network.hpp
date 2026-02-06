#pragma once
#include "GNeuro/Layer.hpp"
#include "GParsing/JSON/GParsing-JSON.hpp"
#include "GNeuro/Activation.hpp"
#include "GNeuro/Random.hpp"
#include "GNeuro/Loss.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace GNeuro {

template <typename value_t> class Network {
private:
  typedef value_t (*activation_t)(value_t _in, bool _derived, std::string &_funcName);
  typedef value_t (*loss_t)(value_t _out, value_t _expected, bool _derived, std::string &_funcName);
  typedef value_t (*mutate_t)(const Network<value_t> &_network);

  std::vector<activation_t> m_activationFunctions {
    GNeuro::None,
    GNeuro::Sigmoid,
    GNeuro::ReLu,
    GNeuro::LeakyReLu,
    GNeuro::TanH
  };

  std::vector<loss_t> m_lossFunctions {
    GNeuro::Error,
    GNeuro::NegativeError,
    GNeuro::SquaredError,
    GNeuro::SquaredNegativeError
  };
public:
  Network() : m_layers(0), m_loss(nullptr) {};
  Network(Network &&) = default;
  Network(const Network &) = default;
  Network &operator=(Network &&) = default;
  Network &operator=(const Network &) = default;
  ~Network() = default;

  void SetLoss(const loss_t _loss) { m_loss = _loss; }
  const loss_t GetLoss() const { return m_loss; }

  void SaveModel(const std::string &_filepath) const {
    GParsing::JSONObject<unsigned char> json;

    GParsing::JSONObject<unsigned char> metadataObject;
    metadataObject.AddMember("version", GParsing::JSONString<unsigned char>("v1"));
    metadataObject.AddMember("type", GParsing::JSONString<unsigned char>("model"));

    json.AddMember("metadata", metadataObject);

    GParsing::JSONValue<unsigned char> lossValue;
    bool found = false;
    for (size_t __lossIndex = 0; __lossIndex < m_lossFunctions.size(); __lossIndex++) {
      if (m_loss == m_lossFunctions[__lossIndex]) {
        std::string funcName;
        m_lossFunctions[__lossIndex](0, 0, false, funcName);
        lossValue.SetString(funcName);
        found = true;
      }
    }

    if (!found) {
      throw std::runtime_error("Cannot serialize loss function.");
    }

    GParsing::JSONArray<unsigned char> weightsArray;
    for (size_t __layerIndex = 0; __layerIndex < GetLayersCount(); __layerIndex++) {
      const auto &layer = operator[](__layerIndex);
      GParsing::JSONArray<unsigned char> layerJSON;

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
        const auto &neuron = layer[__neuronIndex];
        GParsing::JSONArray<unsigned char> neuronJSON;

        for (size_t __weightIndex = 0; __weightIndex < neuron.GetInputsCount(); __weightIndex++) {
          const auto weight = neuron.GetWeight(__weightIndex);
          neuronJSON.PushValue(GParsing::JSONNumber<unsigned char>(weight));
        }

        layerJSON.PushValue(neuronJSON);
      }

      weightsArray.PushValue(layerJSON);
    }

    GParsing::JSONArray<unsigned char> biasesArray;
    for (size_t __layerIndex = 0; __layerIndex < GetLayersCount(); __layerIndex++) {
      const auto &layer = operator[](__layerIndex);
      GParsing::JSONArray<unsigned char> layerJSON;

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
        const auto &neuron = layer[__neuronIndex];
        layerJSON.PushValue(GParsing::JSONNumber<unsigned char>(neuron.GetBias()));
      }

      biasesArray.PushValue(layerJSON);
    }

    GParsing::JSONArray<unsigned char> activationsArray;
    for (size_t __layerIndex = 0; __layerIndex < GetLayersCount(); __layerIndex++) {
      const auto &layer = operator[](__layerIndex);
      GParsing::JSONArray<unsigned char> layerJSON;

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        const auto &neuron = layer[__neuronIndex];

        bool found = false;
        for (size_t __activationIndex = 0; __activationIndex < m_activationFunctions.size(); __activationIndex++) {
          if (neuron.GetActivation() == m_activationFunctions[__activationIndex]) {
            std::string funcName;
            m_activationFunctions[__activationIndex](0, false, funcName);
            layerJSON.PushValue(GParsing::JSONString<unsigned char>(funcName));
            found = true;
          }
        }

        if (!found) {
          throw std::runtime_error(
              "Unknown activation function. Cannot serialize string.");
        }
      }

      activationsArray.PushValue(layerJSON);
    }

    json.AddMember("loss", lossValue);
    json.AddMember("weights", weightsArray);
    json.AddMember("biases", biasesArray);
    json.AddMember("activations", activationsArray);

    if (!json.Serialize(_filepath)) {
      throw std::runtime_error("Cannot serialze JSON.");
    }
  }

  void LoadModel(const std::string &_filepath) {
    GParsing::JSONObject<unsigned char> json;
    if (!json.Parse(_filepath)) {
      throw std::runtime_error("Unable to parse JSON model.");
    }

    const auto &metadataObject = json["metadata"].GetObject();
    const auto &versionString = metadataObject["version"].GetString();
    const auto &typeString = metadataObject["type"].GetString();

    if (typeString != "model") {
      throw std::runtime_error("Unknown JSON type.");
    }

    if (versionString != "v1") {
      throw std::runtime_error("Unknown model version.");
    }

    const auto lossString = json["loss"].GetString();
    bool found = false;
    for (size_t __lossIndex = 0; __lossIndex < m_lossFunctions.size(); __lossIndex++) {
      std::string funcName;
      m_lossFunctions[__lossIndex](0, 0, false, funcName);

      if (lossString == funcName) {
        m_loss = m_lossFunctions[__lossIndex];
        found = true;
      }
    }

    if (!found) {
      throw std::runtime_error("Cannot parse loss function string.");
    }

    ClearLayers();

    const auto &weights = json["weights"].GetArray();
    for (size_t __layerIndex = 0; __layerIndex < weights.GetSize(); __layerIndex++) {
      auto &jsonLayer = weights[__layerIndex].GetArray();
      auto &layer = m_layers.emplace_back(jsonLayer.GetSize());

      for (size_t __neuronIndex = 0; __neuronIndex < jsonLayer.GetSize(); __neuronIndex++) {
        auto &jsonNeuron = jsonLayer[__neuronIndex].GetArray();
        auto &neuron = layer[__neuronIndex];

        neuron.SetInputsCount(jsonNeuron.GetSize());

        for (size_t __weightIndex = 0; __weightIndex < jsonNeuron.GetSize(); __weightIndex++) {
          value_t jsonWeight = jsonNeuron.GetValue(__weightIndex).GetNumber();
          neuron.SetWeight(__weightIndex, jsonWeight);
        }
      }
    }

    const auto &biases = json["biases"].GetArray();
    for (size_t __layerIndex = 0; __layerIndex < biases.GetSize(); __layerIndex++) {
      auto &jsonLayer = biases[__layerIndex].GetArray();
      auto &layer = m_layers[__layerIndex];

      for (size_t __neuronIndex = 0; __neuronIndex < jsonLayer.GetSize(); __neuronIndex++) {
        auto jsonBias = jsonLayer[__neuronIndex].GetNumber();
        auto &neuron = layer[__neuronIndex];

        neuron.SetBias(jsonBias);
      }
    }

    const auto &activation = json["activations"].GetArray();
    for (size_t __layerIndex = 0; __layerIndex < activation.GetSize(); __layerIndex++) {
      auto &jsonLayer = activation[__layerIndex].GetArray();
      auto &layer = m_layers[__layerIndex];

      for (size_t __neuronIndex = 0; __neuronIndex < jsonLayer.GetSize(); __neuronIndex++) {
        auto jsonActivationString = jsonLayer[__neuronIndex].GetString();
        auto &neuron = layer[__neuronIndex];

        bool found = false;
        for (size_t __activationIndex = 0; __activationIndex < m_activationFunctions.size(); __activationIndex++) {
          std::string funcName;
          m_activationFunctions[__activationIndex](0, false, funcName);

          if (jsonActivationString == funcName) {
            neuron.SetActivation(m_activationFunctions[__activationIndex]);
            found = true;
          }
        }

        if (!found) {
          throw std::runtime_error(
              "Could not parse activation function string.");
        }
      }
    }
  };

  [[nodiscard]]
  std::vector<value_t> Calculate(const std::vector<value_t> &_inputs) {
    std::vector<std::vector<value_t>> activatedStructure,
        unactivatedStructure;
    _CalculateOutputStructures(_inputs, activatedStructure, unactivatedStructure);

    return activatedStructure.back();
  };

  void Train(const std::vector<std::vector<value_t>> &_inputsBatch,
             const std::vector<std::vector<value_t>> &_expectedOutputsBatch,
             const value_t _learningRate, const value_t _lossThreshold) {
    if (_inputsBatch.size() != _expectedOutputsBatch.size()) {
      throw std::runtime_error(
          "Inputs and expected outputs batch size doesn't match.");
    }

    value_t meanLoss = _lossThreshold + 1;
    while (meanLoss > _lossThreshold) {
      for (size_t __inputIndex = 0; __inputIndex < _inputsBatch.size();
           __inputIndex++) {
        _BackPropagate(_inputsBatch[__inputIndex],
                       _expectedOutputsBatch[__inputIndex], _learningRate);
      }

      meanLoss = MeanLoss(_inputsBatch, _expectedOutputsBatch);
      std::cout << "Loss: " << meanLoss << std::endl;
    }
  }

  void Mutate(const mutate_t _callback) {
    const value_t mutateAmount = Random(-1.0, 1.0);

    const size_t layerIndex = std::round(Random(0, GetLayersCount() - 1));
    auto &layer = operator[](layerIndex);
    const size_t neuronIndex = std::round(Random(0, layer.GetSize() - 1));
    auto &neuron = layer[neuronIndex];
    const size_t attributeIndex =
        std::round(Random(0, neuron.GetInputsCount()));

    if (attributeIndex <= 0) {
      auto originalValue = neuron.GetBias();
      neuron.SetBias(originalValue + mutateAmount);

      auto reward = _callback(*this);

      neuron.SetBias(originalValue + (mutateAmount * reward));
    } else {
      const size_t weightIndex = attributeIndex - 1;

      auto originalValue = neuron.GetWeight(weightIndex);
      neuron.SetWeight(weightIndex, originalValue + mutateAmount);

      auto reward = _callback(*this);
      neuron.SetWeight(weightIndex, originalValue + (mutateAmount * reward));

      neuron.SetWeight(weightIndex, originalValue + (mutateAmount * reward));
    }
  }

  [[nodiscard]]
  value_t
  MeanLoss(const std::vector<std::vector<value_t>> &_inputsBatch,
           const std::vector<std::vector<value_t>> &_expectedOutputsBatch) {
    value_t meanLoss = 0;

    if (!_HasLossFunction()) {
      throw std::runtime_error("No loss function provided");
    }

    if (_inputsBatch.size() != _expectedOutputsBatch.size()) {
      throw std::runtime_error(
          "Inputs and expected outputs batch size doesn't match.");
    }

    for (size_t __inputIndex = 0; __inputIndex < _inputsBatch.size();
         __inputIndex++) {
      value_t avgLoss = 0;
      const auto &inputs = _inputsBatch[__inputIndex];
      const auto &expectedOutputs = _expectedOutputsBatch[__inputIndex];

      const auto outputs = Calculate(inputs);

      if (outputs.size() != expectedOutputs.size()) {
        throw std::runtime_error(
            "Outputs count doesn't match expected outputs count.");
      }

      for (size_t __outputIndex = 0; __outputIndex < outputs.size(); __outputIndex++) {
        std::string tmp;
        avgLoss += m_loss(outputs[__outputIndex], expectedOutputs[__outputIndex], false, tmp);
      }

      avgLoss /= outputs.size();
      meanLoss += avgLoss;
    }

    meanLoss /= _inputsBatch.size();
    return meanLoss;
  }

  void Randomize() {
    for (size_t __layerIndex = 0; __layerIndex < GetLayersCount();
         __layerIndex++) {
      auto &layer = operator[](__layerIndex);
      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        auto &neuron = layer[__neuronIndex];

        neuron.SetBias(Random(-1.0, 1.0));

        for (size_t __weightIndex = 0; __weightIndex < neuron.GetInputsCount();
             __weightIndex++) {
          neuron.SetWeight(__weightIndex, Random(-1.0, 1.0));
        }
      }
    }
  }

  const size_t GetLayersCount() const { return m_layers.size(); }
  void AddLayer(const Layer<value_t> &_layer) { m_layers.emplace_back(_layer); };
  void AddLayer(const Layer<value_t> &_layer, const activation_t _activation) {
    auto &layer = m_layers.emplace_back(_layer);
    layer.SetActivationFunction(_activation);
  };

  void ClearLayers() { m_layers.clear(); }

  void FitLayers(const size_t _inputCount) {
    if (!_HasLayers()) {
      throw std::runtime_error("No valid layers to fit.");
    }

    auto &layer = operator[](0);
    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
         __neuronIndex++) {
      auto &neuron = layer[__neuronIndex];
      neuron.SetInputsCount(_inputCount);
    }

    for (size_t __layerIndex = 1; __layerIndex < GetLayersCount();
         __layerIndex++) {
      auto &layer = operator[](__layerIndex);
      auto &previousLayer = operator[](__layerIndex - 1);

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        auto &neuron = layer[__neuronIndex];
        neuron.SetInputsCount(previousLayer.GetSize());
      }
    }
  }

  Layer<value_t> &operator[](const size_t _layer) { return m_layers[_layer]; }
  const Layer<value_t> &operator[](const size_t _layer) const {
    return m_layers[_layer];
  }

private:
  void _CalculateOutputStructures(
      const std::vector<value_t> &_inputs,
      std::vector<std::vector<value_t>> &_activatedStructure,
      std::vector<std::vector<value_t>> &_unactivatedStructure) const {
    if (!_HasValidInputs(_inputs.size())) {
      throw std::runtime_error("Inputs amount doesn't fit model.");
    }

    if (!_HasLayers()) {
      throw std::runtime_error("No model layers in network.");
    }

    if (!_HasActivationFunctions()) {
      throw std::runtime_error(
          "Activation functions not completely populated.");
    }

    if (!_HasValidAmountOfWeights()) {
      throw std::runtime_error("Layers not fitted in model.");
    }

    // Layer 0
    const auto &layer = operator[](0);
    auto &activatedLayer = _activatedStructure.emplace_back();
    auto &unactivatedLayer = _unactivatedStructure.emplace_back();
    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
         __neuronIndex++) {
      const auto &neuron = layer[__neuronIndex];
      value_t tmp = 0;

      for (size_t __inputIndex = 0; __inputIndex < _inputs.size();
           __inputIndex++) {
        tmp += _inputs[__inputIndex] * neuron.GetWeight(__inputIndex);
      }

      tmp += neuron.GetBias();

      std::string str;
      activatedLayer.emplace_back(neuron.GetActivation()(tmp, false, str));
      unactivatedLayer.emplace_back(tmp);
    }

    // For layer 1 and onwards
    for (size_t __layerIndex = 1; __layerIndex < GetLayersCount();
         __layerIndex++) {
      const auto &previousLayer = operator[](__layerIndex - 1);
      const auto &layer = operator[](__layerIndex);

      auto &activatedLayer = _activatedStructure.emplace_back();
      auto &unactivatedLayer = _unactivatedStructure.emplace_back();

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        const auto &neuron = layer[__neuronIndex];
        value_t tmp = 0;

        for (size_t __previousNeuronIndex = 0;
             __previousNeuronIndex < previousLayer.GetSize();
             __previousNeuronIndex++) {
          tmp += _activatedStructure[__layerIndex - 1][__previousNeuronIndex] *
                 neuron.GetWeight(__previousNeuronIndex);
        }

        tmp += neuron.GetBias();

        std::string str;
        activatedLayer.emplace_back(neuron.GetActivation()(tmp, false, str));
        unactivatedLayer.emplace_back(tmp);
      }
    }
  }

  void _CalculateGradientStructures(
      const std::vector<std::vector<value_t>> &_activatedStructure,
      const std::vector<std::vector<value_t>> &_unactivatedStructure,
      const std::vector<value_t> &_expectedOutputs,
      std::vector<std::vector<value_t>> &_gradientStructure) {
    if (!_HasLossFunction()) {
      throw std::runtime_error("No loss function provided");
    }

    if (!_HasLayers()) {
      throw std::runtime_error("Empty model");
    }

    _gradientStructure.clear();
    _gradientStructure.resize(GetLayersCount());

    for (size_t __layerIndex = 0; __layerIndex < GetLayersCount();
         __layerIndex++) {
      _gradientStructure[__layerIndex]
          .resize(operator[](__layerIndex).GetSize());
    }

    auto &layer = operator[](GetLayersCount() - 1);
    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
         __neuronIndex++) {
      std::string tmp;
      const auto lossDelta = m_loss(_activatedStructure[GetLayersCount() - 1][__neuronIndex], _expectedOutputs[__neuronIndex], true, tmp);
      const auto activationDelta = layer[__neuronIndex].GetActivation()(_unactivatedStructure[GetLayersCount() - 1][__neuronIndex], true, tmp);

      _gradientStructure[GetLayersCount() - 1][__neuronIndex] = lossDelta * activationDelta;
    }

    for (int64_t __layerIndex = GetLayersCount() - 2; __layerIndex >= 0;
         __layerIndex--) {
      auto &layer = operator[](__layerIndex);
      auto &parentLayer = operator[](__layerIndex + 1);

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        value_t parentDelta = 0;

        for (size_t __parentNeuronIndex = 0;
             __parentNeuronIndex < parentLayer.GetSize();
             __parentNeuronIndex++) {
          parentDelta +=
              _gradientStructure[__layerIndex + 1][__parentNeuronIndex] *
              parentLayer[__parentNeuronIndex].GetWeight(__neuronIndex);
        }

        std::string tmp;
        const auto activationDelta = layer[__neuronIndex].GetActivation()(_unactivatedStructure[__layerIndex][__neuronIndex], true, tmp);

        _gradientStructure[__layerIndex][__neuronIndex] =
            parentDelta * activationDelta;
      }
    }
  }

  void _CalculateGradientStructures(
      const std::vector<value_t> &_inputs,
      const std::vector<value_t> &_expectedOutputs,
      std::vector<std::vector<value_t>> &_gradientStructure) {
    std::vector<std::vector<value_t>> activatedOutputs, unactivateOutputs;
    _CalculateOutputStructures(_inputs, activatedOutputs, unactivateOutputs);
    _CalculateGradientStructures(activatedOutputs, unactivateOutputs,
                                 _expectedOutputs, _gradientStructure);
  }

  void _BackPropagate(const std::vector<value_t> &_inputs,
                      const std::vector<value_t> &_expectedOutputs,
                      const value_t _learningRate) {
    std::vector<std::vector<value_t>> gradientStructure, activatedStructure,
        unactivatedStructure;
    _CalculateOutputStructures(_inputs, activatedStructure,
                               unactivatedStructure);
    _CalculateGradientStructures(activatedStructure, unactivatedStructure,
                                 _expectedOutputs, gradientStructure);

    for (int64_t __layerIndex = GetLayersCount() - 1; __layerIndex >= 0;
         __layerIndex--) {
      auto &layer = operator[](__layerIndex);
      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        auto &neuron = layer[__neuronIndex];

        const auto newBias = neuron.GetBias() - (gradientStructure[__layerIndex][__neuronIndex] * _learningRate);
        neuron.SetBias(newBias);

        for (size_t __weightIndex = 0; __weightIndex < neuron.GetInputsCount();
             __weightIndex++) {
          value_t newWeight;
          if (__layerIndex > 0) {
            newWeight = neuron.GetWeight(__weightIndex) -
                        (gradientStructure[__layerIndex][__neuronIndex] *
                         activatedStructure[__layerIndex - 1][__weightIndex] *
                         _learningRate);
          } else {
            newWeight = neuron.GetWeight(__weightIndex) -
                        (gradientStructure[__layerIndex][__neuronIndex] *
                         _inputs[__weightIndex] * _learningRate);
          }

          neuron.SetWeight(__weightIndex, newWeight);
        }
      }
    }
  }

  // Error checks
  [[nodiscard]]
  bool _HasLayers() const noexcept {
    if (GetLayersCount() <= 0) {
      return false;
    }

    return true;
  }

  [[nodiscard]]
  bool _HasValidInputs(const size_t _inputs) const noexcept {
    const auto &firstLayer = operator[](0);
    for (size_t __neuronIndex = 0; __neuronIndex < firstLayer.GetSize();
         __neuronIndex++) {
      const auto &neuron = firstLayer[__neuronIndex];

      if (neuron.GetInputsCount() != _inputs) {
        return false;
      }
    }

    return true;
  }

  [[nodiscard]]
  bool _HasActivationFunctions() const noexcept {
    for (size_t __layerIndex = 0; __layerIndex < GetLayersCount();
         __layerIndex++) {
      const auto &layer = operator[](__layerIndex);

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        const auto &neuron = layer[__neuronIndex];
        // Neuron activation function check
        if (neuron.GetActivation() == nullptr) {
          return false;
        }
      }
    }

    return true;
  }

  [[nodiscard]]
  bool _HasLossFunction() const noexcept {
    if (!m_loss) {
      return false;
    }

    return true;
  }

  [[nodiscard]]
  bool _HasValidAmountOfWeights() const noexcept {
    for (size_t __layerIndex = 1; __layerIndex < GetLayersCount();
         __layerIndex++) {
      const auto &previousLayer = operator[](__layerIndex - 1);
      const auto &layer = operator[](__layerIndex);

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize();
           __neuronIndex++) {
        const auto &neuron = layer[__neuronIndex];
        // Neuron input weights amount previous layer size check
        if (neuron.GetInputsCount() != previousLayer.GetSize()) {
          return false;
        }
      }
    }

    return true;
  }

  std::vector<Layer<value_t>> m_layers;
  loss_t m_loss;
};
} // namespace GNeuro
