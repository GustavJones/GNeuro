#include "GNeuro/Layer.hpp"

namespace GNeuro {
void Layer::SetActivationFunction(const ACTIVATION_T _func) {
  for (size_t __neuronIndex = 0; __neuronIndex < GetSize(); __neuronIndex++) {
    auto &neuron = operator[](__neuronIndex);

    neuron.SetActivation(_func);
  }
}
}
