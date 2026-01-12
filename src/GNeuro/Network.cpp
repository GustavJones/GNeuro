#include "GNeuro/Network.hpp"
#include "GNeuro/Activation.hpp"
#include "GNeuro/Layer.hpp"
#include "GNeuro/Loss.hpp"
#include "GNeuro/Random.hpp"
#include "GNeuro/Type.hpp"
#include "GParsing/JSON/GParsing-JSON.hpp"
#include "GParsing/JSON/JSONArray.hpp"
#include "GParsing/JSON/JSONString.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace GNeuro {
void Network::SaveModel(const std::string &_filepath) const {
  GParsing::JSONObject<unsigned char> json;

  GParsing::JSONObject<unsigned char> metadataObject;
  metadataObject.AddMember("version", GParsing::JSONString<unsigned char>("v1"));

  json.AddMember("metadata", metadataObject);

  GParsing::JSONValue<unsigned char> lossValue;
  bool found = false;
  for (size_t __lossIndex = 0; __lossIndex < LossFunctions.size(); __lossIndex++) {
    if (m_loss == LossFunctions[__lossIndex].second) {
      lossValue.SetString(LossFunctions[__lossIndex].first);
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

    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      const auto &neuron = layer[__neuronIndex];

      bool found = false;
      for (size_t __activationIndex = 0; __activationIndex < ActivationFunctions.size(); __activationIndex++) {
        if (neuron.GetActivation() == ActivationFunctions[__activationIndex].second) {
          layerJSON.PushValue(GParsing::JSONString<unsigned char>(ActivationFunctions[__activationIndex].first));
          found = true;
        }
      }

      if (!found) {
        throw std::runtime_error("Unknown activation function. Cannot serialize string.");
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

void Network::LoadModel(const std::string &_filepath) {
  GParsing::JSONObject<unsigned char> json;
  if (!json.Parse(_filepath)) {
    throw std::runtime_error("Unable to parse JSON model.");
  }

  const auto &metadataObject = json["metadata"].GetObject();
  const auto &versionString = metadataObject["version"].GetString();

  if (versionString != "v1") {
    throw std::runtime_error("Unknown model version.");
  }

  const auto lossString = json["loss"].GetString();
  bool found = false;
  for (size_t __lossIndex = 0; __lossIndex < LossFunctions.size(); __lossIndex++) {
    if (lossString == LossFunctions[__lossIndex].first) {
      m_loss = LossFunctions[__lossIndex].second;
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
        DECIMAL_T jsonWeight = jsonNeuron.GetValue(__weightIndex).GetNumber();
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
      for (size_t __activationIndex = 0; __activationIndex < ActivationFunctions.size(); __activationIndex++) {
        if (jsonActivationString == ActivationFunctions[__activationIndex].first) {
          neuron.SetActivation(ActivationFunctions[__activationIndex].second);
          found = true;
        }
      }

      if (!found) {
        throw std::runtime_error("Could not parse activation function string.");
      }
    }
  }
}

std::vector<DECIMAL_T> Network::Calculate(const std::vector<DECIMAL_T> &_inputs) {  
  std::vector<std::vector<DECIMAL_T>> activatedStructure, unactivatedStructure;
  _CalculateOutputStructures(_inputs, activatedStructure, unactivatedStructure);

  return activatedStructure.back();
}

void Network::Train(const std::vector<std::vector<DECIMAL_T>> &_inputsBatch, const std::vector<std::vector<DECIMAL_T>> &_expectedOutputsBatch, const DECIMAL_T _learningRate, const DECIMAL_T _lossThreshold) {
  if (_inputsBatch.size() != _expectedOutputsBatch.size()) {
    throw std::runtime_error("Inputs and expected outputs batch size doesn't match.");
  }

  DECIMAL_T meanLoss = _lossThreshold + 1;
  while (meanLoss > _lossThreshold) {
    for (size_t __inputIndex = 0; __inputIndex < _inputsBatch.size(); __inputIndex++) {
      _BackPropagate(_inputsBatch[__inputIndex], _expectedOutputsBatch[__inputIndex], _learningRate);
    }

    meanLoss = MeanLoss(_inputsBatch, _expectedOutputsBatch);
    std::cout << "Loss: " << meanLoss << std::endl;
  }
}

void Network::Mutate(const MUTATE_CALLBACK_T _callback) {
  const DECIMAL_T mutateAmount = Random();
  DECIMAL_T originalValue;

  const size_t layerIndex = std::round(Random(0, GetLayersCount() - 1));
  auto &layer = operator[](layerIndex);
  const size_t neuronIndex = std::round(Random(0, layer.GetSize() - 1));
  auto &neuron = layer[neuronIndex];
  const size_t attributeIndex = std::round(Random(0, neuron.GetInputsCount()));

  if (attributeIndex <= 0) {
    originalValue = neuron.GetBias();
    neuron.SetBias(originalValue + mutateAmount);

    DECIMAL_T reward = _callback();
    neuron.SetBias(originalValue + (mutateAmount * reward));
  } else {
    const size_t weightIndex = attributeIndex - 1;

    originalValue = neuron.GetWeight(weightIndex);
    neuron.SetWeight(weightIndex, originalValue + mutateAmount);

    DECIMAL_T reward = _callback();
    neuron.SetWeight(weightIndex, originalValue + (mutateAmount * reward));
  }
}

DECIMAL_T Network::MeanLoss(const std::vector<std::vector<DECIMAL_T>> &_inputsBatch, const std::vector<std::vector<DECIMAL_T>> &_expectedOutputsBatch) {
  DECIMAL_T meanLoss = 0;
  
  if (!_HasLossFunction()) {
    throw std::runtime_error("No loss function provided");
  }

  if (_inputsBatch.size() != _expectedOutputsBatch.size()) {
    throw std::runtime_error("Inputs and expected outputs batch size doesn't match.");
  }

  for (size_t __inputIndex = 0; __inputIndex < _inputsBatch.size(); __inputIndex++) {
    DECIMAL_T avgLoss = 0;
    const auto &inputs = _inputsBatch[__inputIndex];
    const auto &expectedOutputs = _expectedOutputsBatch[__inputIndex];

    const auto outputs = Calculate(inputs);

    if (outputs.size() != expectedOutputs.size()) {
      throw std::runtime_error("Outputs count doesn't match expected outputs count.");
    }
    
    for (size_t __outputIndex = 0; __outputIndex < outputs.size(); __outputIndex++) {
      avgLoss += m_loss(outputs[__outputIndex], expectedOutputs[__outputIndex], false);
    }

    avgLoss /= outputs.size();
    meanLoss += avgLoss;
  }

  meanLoss /= _inputsBatch.size();
  return meanLoss;
}

void Network::Randomize() {
  for (size_t __layerIndex = 0; __layerIndex < GetLayersCount(); __layerIndex++) {
    auto &layer = operator[](__layerIndex);
    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      auto &neuron = layer[__neuronIndex];

      neuron.SetBias(Random());

      for (size_t __weightIndex = 0; __weightIndex < neuron.GetInputsCount(); __weightIndex++) {
        neuron.SetWeight(__weightIndex, Random());
      }
    }
  }
}

void Network::FitLayers(const size_t _inputCount) {
  if (!_HasLayers()) {
    throw std::runtime_error("No valid layers to fit.");
  }

  auto &layer = operator[](0);
  for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
    auto &neuron = layer[__neuronIndex];
    neuron.SetInputsCount(_inputCount);
  }


  for (size_t __layerIndex = 1; __layerIndex < GetLayersCount(); __layerIndex++) {
    auto &layer = operator[](__layerIndex);
    auto &previousLayer = operator[](__layerIndex - 1);

    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      auto &neuron = layer[__neuronIndex];
      neuron.SetInputsCount(previousLayer.GetSize());
    }
  }
}

void Network::_CalculateOutputStructures(const std::vector<DECIMAL_T> &_inputs, std::vector<std::vector<DECIMAL_T>> &_activatedStructure, std::vector<std::vector<DECIMAL_T>> &_unactivatedStructure) const {
  if (!_HasValidInputs(_inputs.size())) {
    throw std::runtime_error("Inputs amount doesn't fit model.");
  }

  if (!_HasLayers()) {
    throw std::runtime_error("No model layers in network.");
  }

  if (!_HasActivationFunctions()) {
    throw std::runtime_error("Activation functions not completely populated.");
  }

  if (!_HasValidAmountOfWeights()) {
    throw std::runtime_error("Layers not fitted in model.");
  }

  // Layer 0
  const auto &layer = operator[](0);
  auto &activatedLayer = _activatedStructure.emplace_back();
  auto &unactivatedLayer = _unactivatedStructure.emplace_back();
  for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
    const auto &neuron = layer[__neuronIndex];
    DECIMAL_T tmp = 0;
    
    for (size_t __inputIndex = 0; __inputIndex < _inputs.size(); __inputIndex++) {
      tmp += _inputs[__inputIndex] * neuron.GetWeight(__inputIndex);
    }

    tmp += neuron.GetBias();

    activatedLayer.emplace_back(neuron.GetActivation()(tmp, false));
    unactivatedLayer.emplace_back(tmp);
  }

  // For layer 1 and onwards
  for (size_t __layerIndex = 1; __layerIndex < GetLayersCount(); __layerIndex++) {
    const auto &previousLayer = operator[](__layerIndex - 1);
    const auto &layer = operator[](__layerIndex);

    auto &activatedLayer = _activatedStructure.emplace_back();
    auto &unactivatedLayer = _unactivatedStructure.emplace_back();

    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      const auto &neuron = layer[__neuronIndex];
      DECIMAL_T tmp = 0;
      
      for (size_t __previousNeuronIndex = 0; __previousNeuronIndex < previousLayer.GetSize(); __previousNeuronIndex++) {
        tmp += _activatedStructure[__layerIndex - 1][__previousNeuronIndex] * neuron.GetWeight(__previousNeuronIndex);
      }

      tmp += neuron.GetBias();

      activatedLayer.emplace_back(neuron.GetActivation()(tmp, false));
      unactivatedLayer.emplace_back(tmp);
    }
  }
}

void Network::_CalculateGradientStructures(const std::vector<std::vector<DECIMAL_T>> &_activatedStructure, const std::vector<std::vector<DECIMAL_T>> &_unactivatedStructure, const std::vector<DECIMAL_T> &_expectedOutputs, std::vector<std::vector<DECIMAL_T>> &_gradientStructure) {
  if (!_HasLossFunction()) {
    throw std::runtime_error("No loss function provided");
  }

  if (!_HasLayers()) {
    throw std::runtime_error("Empty model");
  }

  _gradientStructure.clear();
  _gradientStructure.resize(GetLayersCount());

  for (size_t __layerIndex = 0; __layerIndex < GetLayersCount(); __layerIndex++) {
    _gradientStructure[__layerIndex].resize(operator[](__layerIndex).GetSize());
  }


  auto &layer = operator[](GetLayersCount() - 1);
  for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
    const DECIMAL_T lossDelta = m_loss(_activatedStructure[GetLayersCount() - 1][__neuronIndex], _expectedOutputs[__neuronIndex], true);
    const DECIMAL_T activationDelta = layer[__neuronIndex].GetActivation()(_unactivatedStructure[GetLayersCount() - 1][__neuronIndex], true);

    _gradientStructure[GetLayersCount() - 1][__neuronIndex] = lossDelta * activationDelta;
  }

  for (int64_t __layerIndex = GetLayersCount() - 2; __layerIndex >= 0; __layerIndex--) {
    auto &layer = operator[](__layerIndex);
    auto &parentLayer = operator[](__layerIndex + 1);

    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      DECIMAL_T parentDelta = 0;

      for (size_t __parentNeuronIndex = 0; __parentNeuronIndex < parentLayer.GetSize(); __parentNeuronIndex++) {
        parentDelta += _gradientStructure[__layerIndex + 1][__parentNeuronIndex] * parentLayer[__parentNeuronIndex].GetWeight(__neuronIndex);
      }

      const DECIMAL_T activationDelta = layer[__neuronIndex].GetActivation()(_unactivatedStructure[__layerIndex][__neuronIndex], true);

      _gradientStructure[__layerIndex][__neuronIndex] = parentDelta * activationDelta;
    }
  }
}

void Network::_CalculateGradientStructures(const std::vector<DECIMAL_T> &_inputs, const std::vector<DECIMAL_T> &_expectedOutputs, std::vector<std::vector<DECIMAL_T>> &_gradientStructure) {
  std::vector<std::vector<DECIMAL_T>> activatedOutputs, unactivateOutputs;
  _CalculateOutputStructures(_inputs, activatedOutputs, unactivateOutputs);
  _CalculateGradientStructures(activatedOutputs, unactivateOutputs, _expectedOutputs, _gradientStructure);
}

void Network::_BackPropagate(const std::vector<DECIMAL_T> &_inputs, const std::vector<DECIMAL_T> &_expectedOutputs, const DECIMAL_T _learningRate) {
  std::vector<std::vector<DECIMAL_T>> gradientStructure, activatedStructure, unactivatedStructure;
  _CalculateOutputStructures(_inputs, activatedStructure, unactivatedStructure);
  _CalculateGradientStructures(activatedStructure, unactivatedStructure, _expectedOutputs, gradientStructure);

  for (int64_t __layerIndex = GetLayersCount() - 1; __layerIndex >= 0; __layerIndex--) {
    auto &layer = operator[](__layerIndex);
    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      auto &neuron = layer[__neuronIndex];

      const DECIMAL_T newBias = neuron.GetBias() - (gradientStructure[__layerIndex][__neuronIndex] * _learningRate);
      neuron.SetBias(newBias);

      for (size_t __weightIndex = 0; __weightIndex < neuron.GetInputsCount(); __weightIndex++) {
        DECIMAL_T newWeight;
        if (__layerIndex > 0) {
          newWeight = neuron.GetWeight(__weightIndex) - (gradientStructure[__layerIndex][__neuronIndex] * activatedStructure[__layerIndex - 1][__weightIndex] * _learningRate);
        }
        else {
          newWeight = neuron.GetWeight(__weightIndex) - (gradientStructure[__layerIndex][__neuronIndex] * _inputs[__weightIndex] * _learningRate);
        }

        neuron.SetWeight(__weightIndex, newWeight);
      }
    }
  }
}

bool Network::_HasLayers() const noexcept {
  if (GetLayersCount() <= 0) {
    return false;
  }

  return true;
}

bool Network::_HasValidInputs(const size_t _inputs) const noexcept {
  const auto &firstLayer = operator[](0);
  for (size_t __neuronIndex = 0; __neuronIndex < firstLayer.GetSize(); __neuronIndex++) {
    const auto &neuron = firstLayer[__neuronIndex];

    if (neuron.GetInputsCount() != _inputs) {
      return false;
    }
  }

  return true;
}

bool Network::_HasActivationFunctions() const noexcept {
  for (size_t __layerIndex = 0; __layerIndex < GetLayersCount(); __layerIndex++) {
    const auto &layer = operator[](__layerIndex);

    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      const auto &neuron = layer[__neuronIndex];
      // Neuron activation function check
      if (neuron.GetActivation() == nullptr) {
        return false;
      }
    }
  }

  return true;
}

bool Network::_HasLossFunction() const noexcept {
  if (!m_loss) {
    return false;
  }

  return true;
}

bool Network::_HasValidAmountOfWeights() const noexcept {
  for (size_t __layerIndex = 1; __layerIndex < GetLayersCount(); __layerIndex++) {
    const auto &previousLayer = operator[](__layerIndex - 1);
    const auto &layer = operator[](__layerIndex);

    for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
      const auto &neuron = layer[__neuronIndex];
      // Neuron input weights amount previous layer size check
      if (neuron.GetInputsCount() != previousLayer.GetSize()) {
        return false;
      }
    }
  }

  return true;
}
} // namespace GNeuro
