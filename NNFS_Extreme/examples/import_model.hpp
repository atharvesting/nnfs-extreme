#include <string>
#include <NN.hpp>

/*
* In order to run this function, use `include "examples/import_model.hpp"` inside NNFS_Extreme.cpp
* and run import_model() inside the main function. Feel free to modify this function to your liking.
*/
void import_model(std::string model_path) {
    auto nn = Network(model_path);
    std::cout << "Network initialized using model binary.\n";

    // Generate an image of the digit 8.
    auto output = nn.feedforward( Matrix<float>(10, 1, std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}) );
    std::cout << "Feedfoward complete!\n";

    // Then we take this output (which is a Matrix<float> containing the pixel densities) and save it as a binary file.
    std::string file_path = "data/output/pixel_data_8.bin";

	std::ofstream pixel_data(file_path, std::ios::binary);
	if (!pixel_data.is_open()) 
		throw std::runtime_error("Could not open pixel_data.bin for writing.");
	
    pixel_data.write( reinterpret_cast<char*>( output.rix.data() ), output.rows * output.cols * sizeof(float) );
    std::cout << "Image data exported! Find it here - " << file_path << std::endl;
}