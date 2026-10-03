#include <cmath> // tanh
#include <algorithm> // transform, max
#include <execution> // execution::par_unseq
#include "activation_functions.hpp"
#include <Spalten/Matrix.hpp>

namespace act {

	float sigmoid(float z) { return 1.0F / (1.0F + std::exp(-z)); }
	float sigmoid_prime(float z) { float s = sigmoid(z); return s * (1.0F - s); }

	float relu(float z) { return std::max(0.0F, z); }
	float relu_prime(float z) { return z > 0.0F ? 1.0F : 0.0F; }

	float tanh(float z) { return std::tanh(z); }
	float tanh_prime(float z) { float t = tanh(z); return 1.0F - (t * t); }

	float leaky_relu(float z) { return z > 0.0F ? z : 0.01F * z; }
	float leaky_relu_prime(float z) { return z > 0.0F ? 1.0F : 0.01F; }

#define ACTIVATION_FUNC_MAT(func) \
	void func(const Matrix<float>& mat, Matrix<float>& dest) { \
		float (*scalar_func)(float) = act::func; \
		act::activation_func_mat(mat, dest, [scalar_func](float z){ return scalar_func(z); }); \
	} \
	Matrix<float> func(const Matrix<float>& mat) { \
		float (*scalar_func)(float) = act::func; \
		return act::activation_func_mat(mat, [scalar_func](float z){ return scalar_func(z); }); \
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

	const Functions& functions(Type type) {
		static constexpr std::array table{
			Functions{sigmoid, sigmoid_prime},
			Functions{relu, relu_prime},
			Functions{tanh, tanh_prime},
			Functions{leaky_relu, leaky_relu_prime},
		};
		return table[static_cast<size_t>(type)];
	}

	const char* name(Type type) {
		static constexpr std::array names{"sigmoid", "relu", "tanh", "leaky_relu"};
		return names[static_cast<size_t>(type)];
	}

}

namespace init {
	float xavier_normal(int fan_in, int fan_out) {
		return std::sqrt(2.0F / static_cast<float>(fan_in + fan_out));
	}

	float lecun_normal(int fan_in, int) {
		return std::sqrt(1.0F / static_cast<float>(fan_in));
	}

	float stddev(Type type, int fan_in, int fan_out) {
		static constexpr std::array<StddevFunction, 2> table{xavier_normal, lecun_normal};
		return table[static_cast<size_t>(type)](fan_in, fan_out);
	}

	const char* name(Type type) {
		static constexpr std::array names{"xavier_normal", "lecun_normal"};
		return names[static_cast<size_t>(type)];
	}

}
