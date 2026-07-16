#include <cmath>
#include <algorithm>

class act {
public:
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
};

