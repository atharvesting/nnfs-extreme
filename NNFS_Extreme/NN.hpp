#include <vector> // vector
#include <ranges> // views::zip
#include "utils.h"
#include "matrix.h"

class Network {
public:
	int num_layers;
	std::vector<int> sizes;
	std::vector<Matrix<float>> biases;
	std::vector<Matrix<float>> weights;

	Network(std::vector<int> nw_sizes)
		: num_layers(nw_sizes.size()), sizes(std::move(nw_sizes)) {
		
		biases.reserve(num_layers - 1);
		weights.reserve(num_layers - 1);
		
		for (size_t i = 1; i < num_layers; i++) {
			int y{ this->sizes[i] };
			biases.emplace_back(mat_random_normal(y, 1));
		}
		for (size_t i = 0; i < num_layers - 1; i++) {
			int x{ this->sizes[i] }, y{ this->sizes[i + 1] };
			weights.emplace_back(mat_random_normal(y, x));
		}
	}

	float feedforward(float a) {
		for (size_t i = 0; i < biases.size(); i++) {
			a = sigmoid(weights[i] * a + biases[i]);
		}
		return a;
	}

	void SGD(auto training_data, int epochs, int min_batch_size, float learning_rate) {

	}

	void update_mini_batch(auto mini_batch, float eta) {
		std::vector<Matrix<float>> nabla_b, nabla_w;
		for ()
	}
};