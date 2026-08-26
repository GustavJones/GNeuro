#include "GNeuro/Network.hpp"
#include "GNeuro/Activation.hpp"
#include "GNeuro/Loss.hpp"

// Example data for an XOR network
// Trains a model to produce the output of an XOR operation
static const GMath::Matrix<double> inputs = {
  {1, 1},
  {1, 0},
  {0, 1},
  {0, 0},
};

static const GMath::Matrix<double> expectedOutputs = {
  {0},
  {1},
  {1},
  {0},
};

int main(int argc, char *argv[]) {
	std::atomic<bool> running = true;

	GNeuro::Network<double> network;
	network.AddLayer(2, GNeuro::LeakyReLu);
	network.AddLayer(2, GNeuro::LeakyReLu);
	network.AddLayer(1, GNeuro::Sigmoid);
	network.CreateModel(GNeuro::SquaredError, 2, true);

	network.Train(inputs, expectedOutputs, 0.1, 0.00001, running);
	// network.Train(inputs, expectedOutputs, 0.05, 10000);
	std::cout << network.Calculate(inputs) << std::endl;
  
  return 0;
}
