#pragma once
#include <iostream>
#include "NN.hpp"
#include "data_loaders.hpp"
#include "utils.hpp"

/*
* In order to run this function, use `include "examples/standard_mnist.hpp"` inside NNFS_Extreme.cpp
* and run standard_mnist() inside the main function. Feel free to modify this function to your liking.
*/
void standard_mnist(int epochs=30, int mini_batch_size=32, float learning_rate=9.0F) {
	std::cout << "Running standard MNIST example!" << std::endl;
	std::cout << "Epochs: " << epochs << ", Mini-batch size: " << mini_batch_size << ", Learning rate: " << learning_rate << std::endl;

	auto training_data = MNIST_loader::load_training_data("data/mnist_train_images.bin", "data/mnist_train_labels.bin", 50000, false);
	std::cout << "Training data loaded.\n";																	// inverted_data = false
	
	auto test_data = MNIST_loader::load_test_data("data/mnist_test_images.bin", "data/mnist_test_labels.bin", 10000);
	std::cout << "Test data loaded.\n";

	auto nn = Network(std::vector<int>{ 784, 30, 10 });
	std::cout << "Network initialized.\n\n";

	nn.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data);

	model_to_file(nn);

}