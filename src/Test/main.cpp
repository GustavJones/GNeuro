#include "GNeuro/Functions.hpp"
#include "GNeuro/GNeuro.hpp"
#include <stdexcept>

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
  GNeuro::Network<double> n;
  GNeuro::Model<double> model;

  GNeuro::Functions<double>::loss_t loss;
  try {
    model.Load("model.json", loss, model.LossFunctionList, model.ActivationFunctionList);
  }
  catch (const std::runtime_error &_e) {
    std::cout << _e.what() << std::endl;
    model.AddLayer(GNeuro::Layer<double>(3, GNeuro::Sigmoid));
    model.AddLayer(GNeuro::Layer<double>(5, GNeuro::Sigmoid));
    model.AddLayer(GNeuro::Layer<double>(1, GNeuro::Sigmoid));
    model.FitLayers(2);
    model.Randomize();
  }


  n.SetModel(model);
  n.SetLoss(GNeuro::SquaredError);

  n.Train(inputs, expectedOutputs, 0.01, 0.001);

  model = n.GetModel();

  model.Save("model.json", n.GetLoss());
  
  return 0;
}
