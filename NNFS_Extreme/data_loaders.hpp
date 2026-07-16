#pragma once
#include <string>
#include <vector>
#include <utility>
#include <Spalten/Matrix.hpp>

using VPM = std::vector<std::pair<Matrix<float>, Matrix<float>>>;

VPM load_training_data(
	const std::string& img_path, const std::string& label_path, size_t num_samples
);

std::vector<std::pair<Matrix<float>, int>> load_test_data(
	const std::string& img_path, const std::string& label_path, size_t num_samples
);