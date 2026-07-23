#include <fstream>
#include <utility>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <Spalten/Matrix.hpp>
#include "data_loaders.hpp"
#include "NN.hpp"

using VPM = std::vector<std::pair<Matrix<float>, Matrix<float>>>;

VPM MNIST_loader::load_training_data(const std::string& img_path,
									 const std::string& label_path, 
									 size_t num_samples, 
									 bool flip_labels) {

	std::ifstream img_file(img_path, std::ios::binary);
	if (!img_file.is_open()) throw std::runtime_error("Could not open " + img_path);
	
	std::ifstream label_file(label_path, std::ios::binary);
	if (!label_file.is_open()) throw std::runtime_error("Could not open " + label_path);

	VPM data;
	data.reserve(num_samples);

	std::vector<float> img_buf(784);
	std::vector<float> label_buf(10);

	for (size_t i = 0; i < num_samples; i++) {
		img_file.read(reinterpret_cast<char*>(img_buf.data()), 784 * sizeof(float));
		label_file.read(reinterpret_cast<char*>(label_buf.data()), 10 * sizeof(float));
		
		Matrix<float> img(784, 1);
		Matrix<float> label(10, 1);

		for (int j = 0; j < 784; j++) img[j] = img_buf[j];
		for (int j = 0; j < 10; j++) label[j] = label_buf[j];

		if (flip_labels == false)
			data.push_back({ std::move(img), std::move(label) });
		else
			data.push_back({ std::move(label), std::move(img) });
	}
	return data;
}

std::vector<std::pair<Matrix<float>, int>> MNIST_loader::load_test_data(
	const std::string& img_path, const std::string& label_path, size_t num_samples) {

	std::ifstream img_file(img_path, std::ios::binary);
	if (!img_file.is_open()) throw std::runtime_error("Could not open " + img_path);
	
	std::ifstream label_file(label_path, std::ios::binary);
	if (!label_file.is_open()) throw std::runtime_error("Could not open " + label_path);

	std::vector<std::pair<Matrix<float>, int>> data;
	data.reserve(num_samples);

	std::vector<float> img_buf(784);
	int label_val;

	for (size_t i = 0; i < num_samples; i++) {
		img_file.read(reinterpret_cast<char*>(img_buf.data()), 784 * sizeof(float));
		label_file.read(reinterpret_cast<char*>(&label_val), sizeof(int));

		Matrix<float> img(784, 1);
		for (int j = 0; j < 784; j++) img[j] = img_buf[j];

		data.push_back({ std::move(img), label_val });
	}
	return data;
}

void Data::export_model(Network& net, const std::string& model_path) {
	std::ofstream model(model_path, std::ios::binary);

	if (!model.is_open()) throw std::runtime_error("Couldn't open model output file.");

	// Number of layers
	model.write( reinterpret_cast<char*>(&net.num_layers), sizeof(size_t) );

	// Neurons per layer
	model.write( reinterpret_cast<char*>(net.sizes.data()), net.num_layers * sizeof(int) );

	// Weights
	for (auto& weight : net.weights) {
		model.write( reinterpret_cast<char*>(&weight.rows), sizeof(size_t));
		model.write( reinterpret_cast<char*>(&weight.cols), sizeof(size_t));
		model.write( reinterpret_cast<char*>(weight.rix.data()), weight.rows * weight.cols * sizeof(float));
	}

	// Biases
	for (auto& bias : net.biases) {
		model.write( reinterpret_cast<char*>(&bias.rows), sizeof(size_t));
		model.write( reinterpret_cast<char*>(&bias.cols), sizeof(size_t));
		model.write( reinterpret_cast<char*>(bias.rix.data()), bias.rows * bias.cols * sizeof(float));
	}
}