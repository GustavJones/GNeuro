#pragma once
#include "GNeuro/Type.hpp"
#include "Layer.hpp"
#include <string>
#include <vector>

namespace GNeuro {
class Network {
public:
  Network() : m_layers(0), m_loss(nullptr) {};
  Network(Network &&) = default;
  Network(const Network &) = default;
  Network &operator=(Network &&) = default;
  Network &operator=(const Network &) = default;
  ~Network() = default;

  void SetLoss(const LOSS_T _loss) { m_loss = _loss; }
  const LOSS_T GetLoss() const { return m_loss; }

  void SaveModel(const std::string &_filepath) const;
  void LoadModel(const std::string &_filepath);

  [[nodiscard]]
  std::vector<DECIMAL_T> Calculate(const std::vector<DECIMAL_T> &_inputs);
  void Train(const std::vector<std::vector<DECIMAL_T>> &_inputsBatch,
             const std::vector<std::vector<DECIMAL_T>> &_expectedOutputsBatch,
             const DECIMAL_T _learningRate, const DECIMAL_T _lossThreshold);
  void Mutate(const MUTATE_CALLBACK_T _callback);

  [[nodiscard]]
  DECIMAL_T
  MeanLoss(const std::vector<std::vector<DECIMAL_T>> &_inputsBatch,
           const std::vector<std::vector<DECIMAL_T>> &_expectedOutputsBatch);

  void Randomize();

  const size_t GetLayersCount() const { return m_layers.size(); }
  void AddLayer(const Layer &_layer) { m_layers.emplace_back(_layer); };
  void AddLayer(const Layer &_layer, const ACTIVATION_T _activation) {
    auto &layer = m_layers.emplace_back(_layer);
    layer.SetActivationFunction(_activation);
  };

  void ClearLayers() { m_layers.clear(); }

  void FitLayers(const size_t _inputCount);

  Layer &operator[](const size_t _layer) { return m_layers[_layer]; }
  const Layer &operator[](const size_t _layer) const {
    return m_layers[_layer];
  }

private:
  void _CalculateOutputStructures(
      const std::vector<DECIMAL_T> &_inputs,
      std::vector<std::vector<DECIMAL_T>> &_activatedStructure,
      std::vector<std::vector<DECIMAL_T>> &_unactivatedStructure) const;
  void _CalculateGradientStructures(
      const std::vector<std::vector<DECIMAL_T>> &_activatedStructure,
      const std::vector<std::vector<DECIMAL_T>> &_unactivatedStructure,
      const std::vector<DECIMAL_T> &_expectedOutputs,
      std::vector<std::vector<DECIMAL_T>> &_gradientStructure);
  void _CalculateGradientStructures(
      const std::vector<DECIMAL_T> &_inputs,
      const std::vector<DECIMAL_T> &_expectedOutputs,
      std::vector<std::vector<DECIMAL_T>> &_gradientStructure);

  void _BackPropagate(const std::vector<DECIMAL_T> &_inputs,
                      const std::vector<DECIMAL_T> &_expectedOutputs,
                      const DECIMAL_T _learningRate);

  // Error checks
  [[nodiscard]]
  bool _HasLayers() const noexcept;

  [[nodiscard]]
  bool _HasValidInputs(const size_t _inputs) const noexcept;

  [[nodiscard]]
  bool _HasActivationFunctions() const noexcept;

  [[nodiscard]]
  bool _HasLossFunction() const noexcept;

  [[nodiscard]]
  bool _HasValidAmountOfWeights() const noexcept;

  std::vector<Layer> m_layers;
  LOSS_T m_loss;
};
} // namespace GNeuro
