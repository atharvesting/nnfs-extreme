#include <cmath> // tanh
#include <algorithm> // transform, max
#include <execution> // execution::par_unseq

#include "activation_functions.hpp"
#include <Spalten/Matrix.hpp>

namespace act {

	float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
	float sigmoid_prime(float z) { float s = sigmoid(z); return s * (1.0f - s); }

	float relu(float z) { return std::max(0.0f, z); }
	float relu_prime(float z) { return z > 0.0f ? 1.0f : 0.0f; }

	float tanh(float z) { return std::tanh(z); }
	float tanh_prime(float z) { float t = tanh(z); return 1.0f - t * t; }

	float leaky_relu(float z) { return z > 0.0f ? z : 0.01f * z; }
	float leaky_relu_prime(float z) { return z > 0.0f ? 1.0f : 0.01f; }

#define ACTIVATION_FUNC_MAT(func) \
	Matrix<float> func(const Matrix<float>& mat) { \
		float (*scalar_func)(float) = act::func; \
		return activation_func_mat(mat, [scalar_func](float z){ return scalar_func(z); }); \
	}
	ACTIVATION_FUNC_MAT(sigmoid)
	ACTIVATION_FUNC_MAT(sigmoid_prime)
	ACTIVATION_FUNC_MAT(relu)
	ACTIVATION_FUNC_MAT(relu_prime)
	ACTIVATION_FUNC_MAT(tanh)
	ACTIVATION_FUNC_MAT(tanh_prime)
	ACTIVATION_FUNC_MAT(leaky_relu)
	ACTIVATION_FUNC_MAT(leaky_relu_prime)
#undef ACTIVATION_FUNC_MAT

}