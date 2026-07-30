#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <Spalten/Matrix.hpp>
#include "NN.hpp"
#include "data_loaders.hpp"

/*
* In order to run this function, use `include "examples/model_export_n_import.hpp"` inside NNFS_Extreme.cpp
* and run model_export_n_import() inside the main function. Feel free to modify this function to your liking.
*/
void model_export_n_import(int epochs = 20, int mini_batch_size = 32, float learning_rate = 2.0F) {
    std::cout << "Running import-export model example!" << std::endl;

	auto training_data = MNIST_loader::load_training_data("data/mnist_train_images.bin", "data/mnist_train_labels.bin", 50000, true);
	std::cout << "Training data loaded.\n";																	// inverted_data = true
	
    // Testing/Evaluation not supported for inverted labels.

	auto nn = Network(std::vector<int>{ 10, 30, 784 });
	std::cout << "Network initialized.\n\n";

	nn.SGD(training_data, epochs, mini_batch_size, learning_rate, {});

    auto model_path = nn.export_model("data/output/", "mnist_inverted");

    // Importing the model weights using a Network constructor
    auto new_nn = Network(model_path);
    std::cout << "Network initialized using model binary.\n";

    // Generate an image of the digit 1.
    auto output = new_nn.feedforward( Matrix<float>(10, 1, std::vector<float>{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}) );
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