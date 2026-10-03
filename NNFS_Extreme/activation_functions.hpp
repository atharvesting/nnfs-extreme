#pragma once

#include <algorithm>
#include <array>
#include <execution>
#include <Spalten/Matrix.hpp>

namespace act {
	using MatrixFunction = void (*)(const Matrix<float>&, Matrix<float>&);
	enum class Type { Sigmoid, Relu, Tanh, LeakyRelu };
	struct Functions { MatrixFunction apply; MatrixFunction derivative; };

	template <typename Func>
	Matrix<float> activation_func_mat(const Matrix<float>& mat, Func func) {
		Matrix<float> res(mat.rows, mat.cols);
		std::transform(
			mat.rix.begin(), mat.rix.end(),
			res.rix.begin(), func
		);
		return res;
	}

	template <typename Func>
	void activation_func_mat(const Matrix<float>& mat, Matrix<float>& dest, Func func) {
		std::transform(
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

	const Functions& functions(Type type);
	const char* name(Type type);
};

namespace init {
	enum class Type { XavierNormal, LeCunNormal };
	using StddevFunction = float (*)(int fan_in, int fan_out);
	float xavier_normal(int fan_in, int fan_out);
	float lecun_normal(int fan_in, int fan_out);
	float stddev(Type type, int fan_in, int fan_out);
	const char* name(Type type);
}
