#pragma once

#include <algorithm>
#include <execution>
#include <Spalten/Matrix.hpp>

namespace act {

	template <typename Func>
	Matrix<float> activation_func_mat(const Matrix<float>& mat, Func func) {
		Matrix<float> res(mat.rows, mat.cols);
		std::transform(
			std::execution::par_unseq,
			mat.rix.begin(), mat.rix.end(),
			res.rix.begin(), func
		);
		return res;
	}

	template <typename Func>
	void activation_func_mat(const Matrix<float>& mat, Matrix<float>& dest, Func func) {
		std::transform(
			std::execution::par_unseq,
			mat.rix.begin(), mat.rix.end(),
			dest.rix.begin(), func
		);
	}

	float sigmoid(float z);
	Matrix<float> sigmoid(const Matrix<float>& mat);
	void sigmoid(const Matrix<float>& mat, Matrix<float>& dest);
	
	float sigmoid_prime(float z);
	Matrix<float> sigmoid_prime(const Matrix<float>& mat);
	void sigmoid_prime(const Matrix<float>& mat, Matrix<float>& dest);

	float tanh(float z);
	Matrix<float> tanh(const Matrix<float>& mat);
	void tanh(const Matrix<float>& mat, Matrix<float>& dest);
	float tanh_prime(float z);
	Matrix<float> tanh_prime(const Matrix<float>& mat);
	void tanh_prime(const Matrix<float>& mat, Matrix<float>& dest);

	float relu(float z);
	Matrix<float> relu(const Matrix<float>& mat);
	void relu(const Matrix<float>& mat, Matrix<float>& dest);
	float relu_prime(float z);
	Matrix<float> relu_prime(const Matrix<float>& mat);
	void relu_prime(const Matrix<float>& mat, Matrix<float>& dest);

	float leaky_relu(float z);
	Matrix<float> leaky_relu(const Matrix<float>& mat);
	void leaky_relu(const Matrix<float>& mat, Matrix<float>& dest);
	float leaky_relu_prime(float z);
	Matrix<float> leaky_relu_prime(const Matrix<float>& mat);
	void leaky_relu_prime(const Matrix<float>& mat, Matrix<float>& dest);
};