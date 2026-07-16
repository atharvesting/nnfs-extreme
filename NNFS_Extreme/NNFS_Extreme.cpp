#include "NN.hpp"
#include "data_loaders.hpp"

int main() {

	auto training_data = load_training_data("data/train_images.bin", "data/train_labels.bin", 50000);
	auto test_data = load_test_data("data/test_images.bin", "data/test_labels.bin", 10000);

	auto nn = Network({ 784, 30, 10 });
	nn.SGD(training_data, 30, 10, 3.0f, test_data);

	return 0;
}