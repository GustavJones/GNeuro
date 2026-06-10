#pragma once
#include "GMath/DynamicArray.hpp"
#include "GMath/Matrix.hpp"
#include "GMath/Types.hpp"
#include "GNeuro/Activation.hpp"
#include "GNeuro/Loss.hpp"
#include "GNeuro/Functions.hpp"
#include "GNeuro/Layer.hpp"
#include "GParsing/JSON/JSONObject.hpp"
#include "GParsing/JSON/JSONArray.hpp"
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace GNeuro {
template <typename value_t> class Model {
private:
  GMath::DynamicArray<typename Functions<value_t>::activation_t> m_activationFunctionList = {
    GNeuro::None,
    GNeuro::Sigmoid,
    GNeuro::ReLu,
    GNeuro::LeakyReLu,
    GNeuro::TanH
  };

  GMath::DynamicArray<typename Functions<value_t>::loss_t> m_lossFunctionList = {
    GNeuro::Error,
    GNeuro::NegativeError,
    GNeuro::SquaredError,
    GNeuro::SquaredNegativeError
  };

  GMath::DynamicArray<Layer<value_t>> m_layers;
  mutable std::mutex m_mutex;

public:
  Model() = default;

  Model(Model &&_m) {
    std::lock_guard<std::mutex> guard(_m.m_mutex);

    m_activationFunctionList = _m.m_activationFunctionList;
    m_lossFunctionList = _m.m_lossFunctionList;
    m_layers = std::move(_m.m_layers);
  }

  Model(const Model &_m) {
    std::lock_guard<std::mutex> guard(_m.m_mutex);

    m_activationFunctionList = _m.m_activationFunctionList;
    m_lossFunctionList = _m.m_lossFunctionList;
    m_layers = _m.m_layers;
  }

  Model &operator=(Model &&_m) {
    if (this == &_m) {
      return *this;
    }

    GMath::DynamicArray<typename Functions<value_t>::activation_t> activationCopy;
    GMath::DynamicArray<typename Functions<value_t>::loss_t> lossCopy;
    GMath::DynamicArray<Layer<value_t>> layersCopy;

    {
      std::lock_guard<std::mutex> guard(_m.m_mutex);
      activationCopy = _m.m_activationFunctionList;
      lossCopy = _m.m_lossFunctionList;
      layersCopy = std::move(_m.m_layers);
    }

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_activationFunctionList = std::move(activationCopy);
      m_lossFunctionList = std::move(lossCopy);
      m_layers = std::move(layersCopy);
    }

    return *this;
  }

  Model &operator=(const Model &_m) {
    if (this == &_m) {
      return *this;
    }

    GMath::DynamicArray<typename Functions<value_t>::activation_t> activationCopy;
    GMath::DynamicArray<typename Functions<value_t>::loss_t> lossCopy;
    GMath::DynamicArray<Layer<value_t>> layersCopy;

    {
      std::lock_guard<std::mutex> guard(_m.m_mutex);
      activationCopy = _m.m_activationFunctionList;
      lossCopy = _m.m_lossFunctionList;
      layersCopy = _m.m_layers;
    }

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_activationFunctionList = std::move(activationCopy);
      m_lossFunctionList = std::move(lossCopy);
      m_layers = std::move(layersCopy);
    }

    return *this;
  }

  ~Model() = default;

  /*
   * Get GNeuro activation function list.
   */
  GMath::DynamicArray<typename Functions<value_t>::activation_t> GetActivationFunctionList() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_activationFunctionList;
  }

  /*
   * Get GNeuro loss function list.
   */
  GMath::DynamicArray<typename Functions<value_t>::loss_t> GetLossFunctionList() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_lossFunctionList;
  }

  /*
   * Get the amount of layers in the model.
   */
  [[nodiscard]]
  GMath::size_t GetLayerCount() const { 
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers.Size(); 
  }

  /*
   * Get the amount of neurons in a layer.
   */
  [[nodiscard]]
  GMath::size_t GetNeuronCount(const GMath::size_t _layerIndex) const { 
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers[_layerIndex].GetSize(); 
  }

  /*
   * Get the amount of weights for a neuron in a layer.
   */
  [[nodiscard]]
  GMath::size_t GetWeightCount(const GMath::size_t _layerIndex) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers[_layerIndex].GetWeightCount();
  }

  /*
   * Get activation function for _neuronIndex from _layerIndex.
   */
  [[nodiscard]]
  typename Functions<value_t>::activation_t GetActivationFunction(const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers[_layerIndex].GetActivationFunction(_neuronIndex);
  }

  /*
   * Set activation function for _neuronIndex from _layerIndex.
   */
  void SetActivationFunction(const typename Functions<value_t>::activation_t _activationFunction, const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex) {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers[_layerIndex].SetActivationFunction(_activationFunction, _neuronIndex);
  }

  /*
   * Get the weight value of a specific neuron in a layer with a specific weight index.
   */
  [[nodiscard]]
  value_t GetWeight(const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex, const GMath::size_t _weightIndex) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers[_layerIndex].GetWeight(_neuronIndex, _weightIndex);
  }

  /*
   * Set the weight for a specific neuron, from a specific layer and weight index.
   */
  void SetWeight(const value_t _value, const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex, const GMath::size_t _weightIndex) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_layers[_layerIndex].SetWeight(_value, _neuronIndex, _weightIndex);
  }

  /*
   * Get the bias value of a specific neuron in a layer.
   */
  [[nodiscard]]
  value_t GetBias(const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_layers[_layerIndex].GetBias(_neuronIndex);
  }

  /*
   * Set the bias for a specific neuron index, from a specific layer.
   */
  void SetBias(const value_t _value, const GMath::size_t _layerIndex, const GMath::size_t _neuronIndex) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_layers[_layerIndex].SetBias(_value, _neuronIndex);
  }

  /*
   * Removes all layers from the model.
   */
  void ClearLayers() { 
    std::lock_guard<std::mutex> guard(m_mutex);
    m_layers.Clear(); 
  }

  /*
   * Add a new layer to the end of the model.
   */
  void AddLayer(const GNeuro::Layer<value_t> &_layer) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_layers.PushBack(_layer);
  }

  /*
   * Update layer weights to fit an input amount and connect each layer's
   * outputs to the next's inputs.
   */
  void FitLayers(const GMath::size_t _inputCount) {
    std::lock_guard<std::mutex> guard(m_mutex);

    if (m_layers.Size() <= 0) {
      throw std::runtime_error("Empty model...");
    }

    m_layers[0].FitLayer(_inputCount);

    for (size_t i = 1; i < m_layers.Size(); i++) {
      m_layers[i].FitLayer(m_layers[i - 1].GetSize());
    }
  }

  /*
   * Fill weights and biases of model with random values.
   */
  void Randomize() {
    std::lock_guard<std::mutex> guard(m_mutex);
    for (size_t i = 0; i < m_layers.Size(); i++) {
      m_layers[i].Randomize();
    }
  }

  /*
   * Calculate the outputs of the network in a GMath::Matrix structure for each
   * layer.
   */
  void Calculate(const GMath::Matrix<value_t> &_inputs, GMath::DynamicArray<GMath::DynamicArray<value_t>> &_unactivatedOutputs, GMath::DynamicArray<GMath::DynamicArray<value_t>> &_activatedOutputs) const {
    std::lock_guard<std::mutex> guard(m_mutex);

    Unlocked_Calculate(_inputs, _unactivatedOutputs, _activatedOutputs);
  }

  /*
   * Calculate the outputs of the network in a GMath::Matrix structure for each
   * layer.
   */
  [[nodiscard]]
  GMath::DynamicArray<GMath::DynamicArray<value_t>> Calculate(const GMath::Matrix<value_t> &_inputs) const {
    GMath::DynamicArray<GMath::DynamicArray<value_t>> _unactivatedOutputs, _activatedOutputs;
    Calculate(_inputs, _unactivatedOutputs, _activatedOutputs);

    return _activatedOutputs;
  }

  /*
   * Calculates the average loss for each input / expected output pair.
   */
  [[nodiscard]]
  value_t MeanLoss(const GMath::Matrix<value_t> &_inputsBatch, const GMath::Matrix<value_t> &_expectedOutputsBatch, const typename Functions<value_t>::loss_t _loss) const {
    value_t meanLoss = 0;

    if (_inputsBatch.Shape().Rows != _expectedOutputsBatch.Shape().Rows) {
      throw std::runtime_error("Inputs and expected outputs batch size doesn't match.");
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (size_t __inputIndex = 0; __inputIndex < _inputsBatch.Shape().Rows; __inputIndex++) {
      const GMath::DynamicArray<value_t> &inputs = _inputsBatch[__inputIndex];
      const GMath::DynamicArray<value_t> &expectedOutputs = _expectedOutputsBatch[__inputIndex];

      meanLoss += Unlocked_Loss(inputs, expectedOutputs, _loss);
    }

    meanLoss /= _inputsBatch.Shape().Rows;
    return meanLoss;
  }

  /*
   * Save the model to a file on disk.
   */
  void Save(const std::string &_filepath, const typename Functions<value_t>::loss_t _loss) {
    const std::string MODEL_SAVE_VERSION = "v1";

    GParsing::JSONObject<unsigned char> json;

    // Metadata
    GParsing::JSONObject<unsigned char> metadata;
    metadata.AddMember("version", (GParsing::JSONString<unsigned char>)MODEL_SAVE_VERSION);
    metadata.AddMember("type", (GParsing::JSONString<unsigned char>)"model");
    json.AddMember("metadata", metadata);

    // Loss
    if (_loss) {
      std::string lossFunctionName;
      _loss(0, 0, false, lossFunctionName);
      json.AddMember("loss", (GParsing::JSONString<unsigned char>)lossFunctionName);
    }
    else {
      throw std::runtime_error("No loss function... Cannot save model.");
    }


    {
      std::lock_guard<std::mutex> guard(m_mutex);
      
      // Weights
      GParsing::JSONArray<unsigned char> weights;
      for (size_t __layerIndex = 0; __layerIndex < m_layers.Size(); __layerIndex++) {
        GParsing::JSONArray<unsigned char> layer;
        for (size_t __neuronIndex = 0; __neuronIndex < m_layers[__layerIndex].GetSize(); __neuronIndex++) {
          GParsing::JSONArray<unsigned char> neuron;
          for (size_t __weightIndex = 0; __weightIndex < m_layers[__layerIndex].GetWeightCount(); __weightIndex++) {
            neuron.PushValue((GParsing::JSONNumber<unsigned char>)m_layers[__layerIndex].GetWeight(__neuronIndex, __weightIndex));
          }  
          layer.PushValue(neuron);
        }
        weights.PushValue(layer);
      }
      json.AddMember("weights", weights);

      // Biases
      GParsing::JSONArray<unsigned char> biases;
      for (size_t __layerIndex = 0; __layerIndex < m_layers.Size(); __layerIndex++) {
        GParsing::JSONArray<unsigned char> layer;
        for (size_t __neuronIndex = 0; __neuronIndex < m_layers[__layerIndex].GetSize(); __neuronIndex++) {
          layer.PushValue((GParsing::JSONNumber<unsigned char>)m_layers[__layerIndex].GetBias(__neuronIndex));
        }
        biases.PushValue(layer);
      }
      json.AddMember("biases", biases);

      // Activation Functions
      GParsing::JSONArray<unsigned char> activations;
      for (size_t __layerIndex = 0; __layerIndex < m_layers.Size(); __layerIndex++) {
        GParsing::JSONArray<unsigned char> layer;
        for (size_t __neuronIndex = 0; __neuronIndex < m_layers[__layerIndex].GetSize(); __neuronIndex++) {
          std::string activationFunctionName;
          typename Functions<value_t>::activation_t activationFunc = m_layers[__layerIndex].GetActivationFunction(__neuronIndex);
          
          if (activationFunc) {
            activationFunc(0, false, activationFunctionName);
          }
          else {
            GNeuro::None(0, false, activationFunctionName);
          }

          layer.PushValue((GParsing::JSONString<unsigned char>)activationFunctionName);
        }
        activations.PushValue(layer);
      }
      json.AddMember("activations", activations);
    }
   
    if (!json.Serialize(_filepath)) {
      throw std::runtime_error("Error serializing model to JSON.");
    }
  }

  /*
   * Load the model from a file on disk.
   */
  void Load(const std::string &_filepath, typename Functions<value_t>::loss_t &_loss, const GMath::DynamicArray<typename Functions<value_t>::loss_t> &_availableLossFunctions, const GMath::DynamicArray<typename Functions<value_t>::activation_t> &_availableActivationFunctions) {
    GParsing::JSONObject<unsigned char> json;
    if (!json.Parse(_filepath)) {
      throw std::runtime_error("Error parsing model from JSON.");
    }

    const auto &metadataObject = json["metadata"].GetObject();
    const auto &versionString = metadataObject["version"].GetString();
    const auto &typeString = metadataObject["type"].GetString();

    if (typeString != "model") {
      throw std::runtime_error("Unknown JSON type.");
    }

    if (versionString == "v1") {
      LoadV1(json, _loss, _availableLossFunctions, _availableActivationFunctions);
    }
    else {
      throw std::runtime_error("Unknown model version");
    }
  }

private:
  /*
   * Calculate the loss of an input / expected output pair.
   * Not thread safe.
   */
  [[nodiscard]]
  value_t Unlocked_Loss(const GMath::Matrix<value_t> &_inputs, const GMath::Matrix<value_t> &_expectedOutputs, const typename Functions<value_t>::loss_t _loss) const {
    value_t avgLoss = 0;

    GMath::DynamicArray<GMath::DynamicArray<value_t>> uOutputs, aOutputs;

    Unlocked_Calculate(_inputs, uOutputs, aOutputs);

    if (aOutputs.Size() <= 0) {
      throw std::runtime_error("Unknown error when calculating outputs...");
    }

    const auto outputs = aOutputs[aOutputs.Size() - 1];

    if (outputs.Size() != _expectedOutputs.Shape().Columns) {
      throw std::runtime_error("Outputs count doesn't match expected outputs count.");
    }

    for (size_t __outputIndex = 0; __outputIndex < outputs.Size(); __outputIndex++) {
      std::string _;
      avgLoss += _loss(outputs[__outputIndex], _expectedOutputs[0][__outputIndex], false, _);
    }

    avgLoss /= outputs.Size();

    return avgLoss;
  }


  /*
   * Calculate the outputs of the network in a GMath::Matrix structure for each
   * layer.
   */
  void Unlocked_Calculate(const GMath::Matrix<value_t> &_inputs, GMath::DynamicArray<GMath::DynamicArray<value_t>> &_unactivatedOutputs, GMath::DynamicArray<GMath::DynamicArray<value_t>> &_activatedOutputs) const {
    if (m_layers.Size() <= 0) {
      throw std::runtime_error("Empty model...");
    }

    GMath::Matrix<value_t> unactivatedOutputs, activatedOutputs;
    _unactivatedOutputs.Clear();
    _activatedOutputs.Clear();

    const Layer<value_t> &firstLayer = m_layers[0];
    firstLayer.Calculate(_inputs, unactivatedOutputs, activatedOutputs);
    _unactivatedOutputs.PushBack(unactivatedOutputs[0]);
    _activatedOutputs.PushBack(activatedOutputs[0]);

    for (size_t i = 1; i < m_layers.Size(); i++) {
      m_layers[i].Calculate(activatedOutputs, unactivatedOutputs, activatedOutputs);
      _unactivatedOutputs.PushBack(unactivatedOutputs[0]);
      _activatedOutputs.PushBack(activatedOutputs[0]);
    }
  }

  /*
   * Load a V1 save file from disk.
   */
  void LoadV1(const GParsing::JSONObject<unsigned char> &_json, typename Functions<value_t>::loss_t &_loss, const GMath::DynamicArray<typename Functions<value_t>::loss_t> &_availableLossFunctions, const GMath::DynamicArray<typename Functions<value_t>::activation_t> &_availableActivationFunctions) {
    const auto lossString = _json["loss"].GetString();
    bool found = false;
    for (size_t __lossIndex = 0; __lossIndex < _availableLossFunctions.Size(); __lossIndex++) {
      std::string funcName;
      _availableLossFunctions[__lossIndex](0, 0, false, funcName);

      if (lossString == funcName) {
        _loss = _availableLossFunctions[__lossIndex];
        found = true;
      }
    }

    if (!found) {
      throw std::runtime_error("Cannot parse loss function string.");
    }

    Model<value_t> modelCopy;

    const auto &weights = _json["weights"].GetArray();
    const auto &biases = _json["biases"].GetArray();
    const auto &activation = _json["activations"].GetArray();

    if (weights.GetSize() != biases.GetSize() || weights.GetSize() != activation.GetSize()) {
      throw std::runtime_error("Malformed json model");
    }

    for (size_t __layerIndex = 0; __layerIndex < weights.GetSize(); __layerIndex++) {
      auto &jsonWeightsLayer = weights[__layerIndex].GetArray();
      auto &jsonBiasesLayer = biases[__layerIndex].GetArray();
      auto &jsonActivationLayer = activation[__layerIndex].GetArray();

      if (jsonWeightsLayer.GetSize() <= 0) {
        continue;
      }

      Layer<value_t> layer(jsonWeightsLayer.GetSize());
      layer.FitLayer(jsonWeightsLayer[0].GetArray().GetSize());

      for (size_t __neuronIndex = 0; __neuronIndex < layer.GetSize(); __neuronIndex++) {
        auto &jsonWeight = jsonWeightsLayer[__neuronIndex].GetArray();
        auto jsonBias = jsonBiasesLayer[__neuronIndex].GetNumber();
        auto jsonActivation = jsonActivationLayer[__neuronIndex].GetString();

        layer.SetBias(jsonBias, __neuronIndex);

        for (size_t __weightIndex = 0; __weightIndex < layer.GetWeightCount(); __weightIndex++) {
          value_t weight = jsonWeight.GetValue(__weightIndex).GetNumber();
          layer.SetWeight(weight, __neuronIndex, __weightIndex);
        }

        bool found = false;
        for (size_t __activationIndex = 0; __activationIndex < _availableActivationFunctions.Size(); __activationIndex++) {
          std::string funcName;
          _availableActivationFunctions[__activationIndex](0, false, funcName);

          if (jsonActivation == funcName) {
            layer.SetActivationFunction(_availableActivationFunctions[__activationIndex], __neuronIndex);
            found = true;
          }
        }

        if (!found) {
          layer.SetActivationFunction(GNeuro::None, __neuronIndex);
        }
      }

      modelCopy.AddLayer(layer);
    }

    GMath::DynamicArray<typename Functions<value_t>::activation_t> activationCopy;
    GMath::DynamicArray<typename Functions<value_t>::loss_t> lossCopy;

    {
      std::lock_guard<std::mutex> guard(m_mutex);
      activationCopy = m_activationFunctionList;
      lossCopy = m_lossFunctionList;
    }

    modelCopy.m_activationFunctionList = activationCopy;
    modelCopy.m_lossFunctionList = lossCopy;

    *this = modelCopy;
  }
};
} // namespace GNeuro
