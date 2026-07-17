#pragma once

#include <Spalten/Matrix.hpp>

namespace act {

	template <typename Func>
	Matrix<float> activation_func_mat(const Matrix<float>& mat, Func func) {
		Matrix<float> result(mat.rows, mat.cols);
		std::transform(
			std::execution::par_unseq,
			mat.rix.begin(), mat.rix.end(),
			result.rix.begin(), func
		);
		return result;
	}

	float sigmoid(float z);
	Matrix<float> sigmoid(const Matrix<float>& mat);
	float sigmoid_prime(float z);
	Matrix<float> sigmoid_prime(const Matrix<float>& mat);

	float tanh(float z);
	Matrix<float> tanh(const Matrix<float>& mat);
	float tanh_prime(float z);
	Matrix<float> tanh_prime(const Matrix<float>& mat);

	float relu(float z);
	Matrix<float> relu(const Matrix<float>& mat);
	float relu_prime(float z);
	Matrix<float> relu_prime(const Matrix<float>& mat);

	float leaky_relu(float z);
	Matrix<float> leaky_relu(const Matrix<float>& mat);
	float leaky_relu_prime(float z);
	Matrix<float> leaky_relu_prime(const Matrix<float>& mat);
};