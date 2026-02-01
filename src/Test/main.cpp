#include <iostream>
#include <vector>
#include "GNeuro/GNeuro.hpp"

// Example data for an XOR network
// Trains a model to produce the output of an XOR operation
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

int main(int argc, char *argv[]) {
  // Create a network object.
  GNeuro::Network network;

  try {
    // Try to load a model file if it exists, otherwise create a new model.
    network.LoadModel("model.json");
  }
  catch (const std::exception &e) {
    // Specify the new model's properties
  
    // Set the model loss function
    network.SetLoss(GNeuro::SquaredError);

    // Add neuron layers
    network.AddLayer(GNeuro::Layer(2), GNeuro::Sigmoid);
    network.AddLayer(GNeuro::Layer(1), GNeuro::Sigmoid);

    // Add layer weights by fitting to model size and input count
    network.FitLayers(2);

    // Randomize model's weights and bias values
    network.Randomize();

    // Train the model with a learning rate of 0.01 until a loss threshold of 0.001 is reached
    network.Train(inputs, expectedOutputs, 0.01, 0.001);
  }  

  // Feed inputs through the network and store outputs
  auto output = network.Calculate(inputs[0]);
  std::cout << output[0] << std::endl;

  // Save model to a JSON file
  network.SaveModel("model.json");

  return 0;
}
