#include <cmath>
#include <algorithm>
#include <execution>

#include <Spalten/Matrix.hpp>

class act {
public:
	static inline Matrix<float> activation_func_mat(const Matrix<float>& mat, auto func) {
		Matrix<float> result(mat.rows, mat.cols);
		std::transform(
			std::execution::par_unseq,
			mat.rix.begin(), mat.rix.end(),
			result.rix.begin(), func
		);
		return result;
	}

	static inline float sigmoid(float z) {
		return 1.0f / (1.0f + std::exp(-z));
	}
	static inline float sigmoid_prime(float z) {
		float s = sigmoid(z);
		return s * (1.0f - s);
	}

	static inline float relu(float z) {
		return std::max(0.0f, z);
	}
	static inline float relu_prime(float z) {
		return z > 0.0f ? 1.0f : 0.0f;
	}

	static inline float tanh(float z) {
		return std::tanh(z);
	}
	static inline float tanh_prime(float z) {
		float t = tanh(z);
		return 1.0f - t * t;
	}

	static inline float leaky_relu(float z) {
		return z > 0.0f ? z : 0.01f * z;
	}
	static inline float leaky_relu_prime(float z) {
		return z > 0.0f ? 1.0f : 0.01f;
	}

#define ACTIVATION_FUNC_MAT(func) \
	static inline Matrix<float> func (const Matrix<float>& mat) { \
		return activation_func_mat(mat, [](float z){ return act::func(z); }); \
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

};

