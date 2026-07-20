#include <algorithm>
#include <fstream>
#include "NN.hpp"
#include "data_loaders.hpp"

int main() {

	auto training_data = MNIST_loader::load_training_data("data/train_images.bin", 
														  "data/train_labels.bin", 
														  50000, true);
	std::cout << "Training data loaded.\n";
	
	// auto test_data = MNIST_loader::load_test_data("data/test_images.bin", "data/test_labels.bin", 10000);
	// std::cout << "Test data loaded.\n";

	auto nn = Network({ 10, 30, 784 });
	std::cout << "Network initialized.\n\n";

	auto weights_and_biases = nn.SGD(training_data, 30, 10, 2.0f, {});
	Matrix<float> output = nn.feedforward( Matrix<float>(10, 1, std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}) );

	std::ofstream pixel_data("data/output/pixel_data.bin", std::ios::binary);
	if (!pixel_data.is_open()) 
		throw std::runtime_error("Could not open pixel_data.bin for writing.");
	pixel_data.write( reinterpret_cast<char*>( output.rix.data() ), output.rows * output.cols * sizeof(float) );
	return 0;
}

// 1. Train the neural network.
// 2. Save the best set of weights and biases to a file.
// 3. Feedforward some digit data and get the pixel intensity matrix as output.
// 4. Save the output matrix to a file.
// 5. Load the output matrix using Python and plot it using matplotlib to visualize the digit.