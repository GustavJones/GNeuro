#include "GNeuro/GNeuro.hpp"

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
	GNeuro::Model<double> model;

	try {
		model.Load("model.json", GNeuro::FunctionType<double>::GetLossFunctions(), GNeuro::FunctionType<double>::GetActivationFunctions());
		network.SetModel(model);
	} catch (...) {
		model.AddLayer(2, GNeuro::LeakyReLu);
		model.AddLayer(1, GNeuro::Sigmoid);
		model.SetLossFunction(GNeuro::SquaredError);
		model.Fit(2);
		model.Randomize();
		network.SetModel(model);
	}

	network.Train(inputs, expectedOutputs, 0.1, 0.001, running);
	// network.Train(inputs, expectedOutputs, 0.05, 10000);
	std::cout << network.Calculate(inputs) << std::endl;

	network.GetModel().Save("model.json");

  return 0;
}
