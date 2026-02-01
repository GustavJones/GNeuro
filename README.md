# GNeuro

## Table of contents:

1. [Description](#description)
2. [Usage](#usage)

## Description

### What is GNeuro?

A simple library to create and train neural networks of developer specified size.

### Why write my own library?

It is a simple library that I wrote to teach myself how neural networks work and to figure out
how backpropagation works on a fundamental level.

### Why use my own sub-library for JSON?

To keep the library under a permissive license that I control and to keep the project from scratch.

## Usage

> [!NOTE]
> This library uses the `double_t` type from the `cmath` include as the numerical type. If you need another type it can be changed from `include/GNeuro/Type.hpp` file.

> [!NOTE]
> An example can be seen in the `Test` source project included in this library. 

```c++
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
```
