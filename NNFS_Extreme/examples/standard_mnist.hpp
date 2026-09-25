#pragma once
#include <iostream>
#include "NN.hpp"

/*
* In order to run this function, use `include "examples/standard_mnist.hpp"` inside NNFS_Extreme.cpp
* and run standard_mnist() inside the main function. Feel free to modify this function to your liking.
*/
void standard_mnist(Config& config) {
    std::string thread_state;
    config.thread_state == Threading::Single ? thread_state = "Single" : thread_state = "Multi";
	std::cout << "Running standard MNIST example!" << std::endl;
	std::cout << "Epochs: " << config.epochs
              << ", Mini-batch size: " << config.mini_batch_size
              << ", Learning rate: " << config.eta
              << ", Threading: " << thread_state
              << std::endl;

	auto nn = Network(config);
	std::cout << "Network initialized.\n\n";

	nn.SGD();
}
