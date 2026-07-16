#include <fstream>
#include <utility>
#include <string>
#include <vector>

#include <Spalten/Matrix.hpp>

using VPM = std::vector<std::pair<Matrix<float>, Matrix<float>>>;

VPM load_training_data(
	const std::string& img_path, const std::string& label_path, size_t num_samples) {

	std::ifstream img_file(img_path, std::ios::binary);
	std::ifstream label_file(label_path, std::ios::binary);

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

		data.push_back({ std::move(img), std::move(label) });
	}
	return data;
}

std::vector<std::pair<Matrix<float>, int>> load_test_data(
	const std::string& img_path, const std::string& label_path, size_t num_samples) {

	std::ifstream img_file(img_path, std::ios::binary);
	std::ifstream label_file(label_path, std::ios::binary);

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