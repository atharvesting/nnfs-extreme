
// void standard_mnist() {

// }

// #include <algorithm>
// #include <fstream>
// #include "NN.hpp"
// #include "data_loaders.hpp"

// int main() {

// 	auto training_data = MNIST_loader::load_training_data("data/train_images.bin", 
// 														  "data/train_labels.bin", 
// 														  50000, true);
// 	std::cout << "Training data loaded.\n";
	
// 	// auto test_data = MNIST_loader::load_test_data("data/test_images.bin", "data/test_labels.bin", 10000);
// 	// std::cout << "Test data loaded.\n";

// 	auto nn = Network(std::vector<int>{ 10, 30, 784 });
// 	std::cout << "Network initialized.\n\n";

// 	nn.SGD(training_data, 20, 10, 2.0F, {});

// 	Matrix<float> output = nn.feedforward( Matrix<float>(10, 1, std::vector<float>{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}) );

// 	std::cout << "Exporting feedforward image data\n";
// 	std::ofstream pixel_data("data/output/pixel_data_0.bin", std::ios::binary);
// 	if (!pixel_data.is_open()) 
// 		throw std::runtime_error("Could not open pixel_data.bin for writing.");
// 	pixel_data.write( reinterpret_cast<char*>( output.rix.data() ), output.rows * output.cols * sizeof(float) );

// 	std::cout << "Exporting model...\n";
// 	Data::export_model(nn, "data/output/model.bin");
// 	std::cout << "Model exported successfully!\n";

// 	auto newNN = Network("data/output/model.bin");
// 	std::cout << "Network initialized from exported parameters!\n";

// 	auto output2 = newNN.feedforward( Matrix<float>(10, 1, std::vector<float>{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}) );

// 	// Matrix<float> output = nn.feedforward( Matrix<float>(10, 1, std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}) );

// 	std::cout << "Exporting feedforward image data\n";
// 	std::ofstream pixel_data2("data/output/pixel_data_fromExported_0.bin", std::ios::binary);
// 	if (!pixel_data2.is_open()) 
// 		throw std::runtime_error("Could not open pixel_data.bin for writing.");
// 	pixel_data2.write( reinterpret_cast<char*>( output2.rix.data() ), output2.rows * output2.cols * sizeof(float) );


// 	return 0;
// }