#pragma once
#include <iostream>
#include <fstream>
#include "NN.hpp"
#include "data_loaders.hpp"

/*
* In order to run this function, use `include "examples/inverted_mnist.hpp"` inside NNFS_Extreme.cpp
* and run inverted_mnist() inside the main function. Feel free to modify this function to your liking.
*/
void inverted_mnist(int epochs=20, int mini_batch_size=10, float learning_rate=2.0F) {
    std::cout << "Running inverted MNIST example!" << std::endl;

	auto training_data = MNIST_loader::load_training_data("data/mnist_train_images.bin", "data/mnist_train_labels.bin", 50000, true);
	std::cout << "Training data loaded.\n";														            // inverted_data = true
	
	// Testing is not yet supported for digit to image data.

	auto nn = Network(std::vector<int>{ 10, 30, 784 }); // This topology is surprisingly effective.
	std::cout << "Network initialized.\n\n";

	nn.SGD(training_data, epochs, mini_batch_size, learning_rate, {});
    std::cout << "Training complete!\n";

    // For now, we try a feedforward with the trained parameters to evaluate the output.
	Matrix<float> output = nn.feedforward( Matrix<float>(10, 1, std::vector<float>{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}) );  // Generate an image of 0.
    std::cout << "Feedfoward complete!\n";

    // Then we take this output (which is a Matrix<float> containing the pixel densities) and save it as a binary file.
    std::string file_path = "data/output/pixel_data.bin";

	std::ofstream pixel_data(file_path, std::ios::binary);
	if (!pixel_data.is_open()) 
		throw std::runtime_error("Could not open pixel_data.bin for writing.");
	
    pixel_data.write( reinterpret_cast<char*>( output.rix.data() ), output.rows * output.cols * sizeof(float) );
    std::cout << "Image data exported! Find it here - " << file_path << std::endl;

    /* !!! IMPORTANT !!!
    * In order to view the image, run the python script here (scripts/pixelarray_to_image.py) using the specified filepath.
    */
}