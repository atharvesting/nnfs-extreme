#pragma once

#include <Spalten/Matrix.hpp>

namespace act {

	template <typename Func>
	void activation_func_mat(const Matrix<float>& mat, Matrix<float>& dest, Func func) {
		std::transform(
			std::execution::par_unseq,
			mat.rix.begin(), mat.rix.end(),
			dest.rix.begin(), func
		);
	}

	float sigmoid(float z);
	void sigmoid(const Matrix<float>& mat, Matrix<float>& dest);
	float sigmoid_prime(float z);
	void sigmoid_prime(const Matrix<float>& mat, Matrix<float>& dest);

	float tanh(float z);
	void tanh(const Matrix<float>& mat, Matrix<float>& dest);
	float tanh_prime(float z);
	void tanh_prime(const Matrix<float>& mat, Matrix<float>& dest);

	float relu(float z);
	void relu(const Matrix<float>& mat, Matrix<float>& dest);
	float relu_prime(float z);
	void relu_prime(const Matrix<float>& mat, Matrix<float>& dest);

	float leaky_relu(float z);
	void leaky_relu(const Matrix<float>& mat, Matrix<float>& dest);
	float leaky_relu_prime(float z);
	void leaky_relu_prime(const Matrix<float>& mat, Matrix<float>& dest);
};