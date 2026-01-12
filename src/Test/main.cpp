#include <iostream>
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
  try {
    network.LoadModel("model.json");
  }
  catch (const std::exception &e) {
    network.SetLoss(GNeuro::SquaredError);

    network.AddLayer(GNeuro::Layer(2), GNeuro::Sigmoid);
    network.AddLayer(GNeuro::Layer(1), GNeuro::Sigmoid);

    network.FitLayers(2);

    network.Randomize();
    network.Train(inputs, expectedOutputs, 0.01, 0.001);
  }  

  auto output = network.Calculate(inputs[0]);
  std::cout << output[0] << std::endl;

  network.SaveModel("model.json");

  return 0;
}
