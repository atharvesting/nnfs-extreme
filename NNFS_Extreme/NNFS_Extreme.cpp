#include <algorithm>
#include <fstream>
#include "NN.hpp"
#include "data_loaders.hpp"

int main() {
	auto newNN = Network("data/output/zzmnist_10-30-784_ep20-lr2.bin");
	std::cout << "Model imported!\n";
	newNN.export_model("data/output/", "mnist");
	std::cout << "Named Model exported!" << std::endl;

	// auto output2 = newNN.feedforward( Matrix<float>(10, 1, std::vector<float>{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}) );

	// // Matrix<float> output = nn.feedforward( Matrix<float>(10, 1, std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}) );

	// std::cout << "Exporting feedforward image data\n";
	// std::ofstream pixel_data2("data/output/pixel_data_fromExported_2.bin", std::ios::binary);
	// if (!pixel_data2.is_open()) 
	// 	throw std::runtime_error("Could not open pixel_data.bin for writing.");
	// pixel_data2.write( reinterpret_cast<char*>( output2.rix.data() ), output2.rows * output2.cols * sizeof(float) );


	return 0;
}