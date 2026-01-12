#include <vector>
#include "GNeuro/GNeuro.hpp"

static const std::vector<std::vector<GNeuro::DECIMAL_T>> inputs = {
  {1, 1},
  {1, 0},
  {0, 1},
  {0, 0},
};

static const std::vector<std::vector<GNeuro::DECIMAL_T>> expectedOutputs = {
  {0},
  {1},
  {1},
  {0},
};

GNeuro::DECIMAL_T Callback() {
  return 0;
}

int main(int argc, char *argv[]) {
  GNeuro::Network network;
  network.SetLoss(GNeuro::SquaredError);

  network.AddLayer(GNeuro::Layer(3), GNeuro::LeakyReLu);
  network.AddLayer(GNeuro::Layer(3), GNeuro::TanH);
  network.AddLayer(GNeuro::Layer(1), GNeuro::Sigmoid);

  network.FitLayers(2);

  network.Randomize();
  network.Train(inputs, expectedOutputs, 0.01, 0.0001);

  return 0;
}
