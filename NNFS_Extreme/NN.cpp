#include <vector> // vector
#include <assert.h> // assert
#include <random> // shuffle
#include <ranges> // views::zip
#include <algorithm> // max_element
#include <Spalten/Matrix.hpp>
#include "activation_functions.hpp"

class Network {
public:
	int num_layers;
	std::vector<int> sizes;
	std::vector<Matrix<float>> biases;
	std::vector<Matrix<float>> weights;

	Network(std::vector<int> nw_sizes)
		: num_layers(nw_sizes.size()), sizes(std::move(nw_sizes)) {

		assert(sizes.size() > 1, "Network must have at least two layers.");

		biases.reserve(num_layers - 1);
		weights.reserve(num_layers - 1);

		for (size_t i = 1; i < num_layers; i++) {
			int y{ sizes[i] };
			biases.emplace_back(mat_random_normal(y, 1));
		}
		for (size_t i = 0; i < num_layers - 1; i++) {
			int x{ sizes[i] }, y{ sizes[i + 1] };
			// Left layer has x neurons, right layer has y neurons, so the weight matrix is y rows by x columns
			// This makes it multipliable with the left layer's output vector (x rows by 1 column)
			weights.emplace_back(mat_random_normal(y, x));
		}
	}

	template <typename T>
	auto feedforward(Matrix<T> a) const {
		for (size_t i = 0; i < biases.size(); i++) {
			a = act::sigmoid(weights[i] * a + biases[i]);
		}
		return a;
	}

	void update_mini_batch(auto mini_batch, float eta) {

		size_t num_param_layers{ num_layers - 1 };

		std::vector<Matrix<float>> nabla_b(num_param_layers), nabla_w(num_param_layers);

		for (size_t i = 0; i < num_param_layers; i++) {
			nabla_b[i] = Matrix<float>::zeros(biases[i].rows, biases[i].cols);
			nabla_w[i] = Matrix<float>::zeros(weights[i].rows, weights[i].cols);
		}

		std::vector<Matrix<float>> delta_nabla_b = nabla_b;
		std::vector<Matrix<float>> delta_nabla_w = nabla_w;

		for (const auto& [x, y] : mini_batch) {
			backprop(x, y, delta_nabla_b, delta_nabla_w);

			for (size_t i = 0; i < num_param_layers; i++) {
				for (size_t j = 0; j < nabla_b[i].size(); j++) {
					nabla_b[i][j] += delta_nabla_b[i][j];
				}
				for (size_t j = 0; j < nabla_w[i].size(); j++) {
					nabla_w[i][j] += delta_nabla_w[i][j];
				}
			}

			for (size_t i = 0; i < num_param_layers; i++) {
				std::fill(delta_nabla_b[i].rix.begin(), delta_nabla_b[i].rix.end(), 0.0f);
				std::fill(delta_nabla_w[i].rix.begin(), delta_nabla_w[i].rix.end(), 0.0f);
			}
		}
		const double scale = eta / static_cast<float>(mini_batch.size());

		for (size_t i = 0; i < num_param_layers; i++) {
			for (size_t j = 0; j < weights[i].size(); j++) {
				weights[i][j] -= scale * nabla_w[i][j];
			}
			for (size_t j = 0; j < biases[i].size(); j++) {
				biases[i][j] -= scale * nabla_b[i][j];
			}
		}
	}

	void SGD(
		auto training_data, int epochs,
		int min_batch_size, float learning_rate,
		auto test_data
	) {

		size_t n_test = test_data.size();

		int n = training_data.size();

		for (int j = 0; j < epochs; j++) {
			std::shuffle(training_data.begin(), training_data.end(),
				std::mt19937{ std::random_device{}() });

			std::vector<std::vector<decltype(training_data)::value_type>> mini_batches;
			for (size_t k = 0; k <= n; k += min_batch_size) {
				using DataType = decltype(training_data)::value_type;
				mini_batches.push_back(
					std::vector<DataType>(training_data.begin() + k,
						training_data.begin() + k + min_batch_size)
				);
			}
			std::cout << std::format("Epoch {}: {} / {}", j, evaluate(test_data), n_test) << std::endl;
		}
	}

	void backprop(auto x, auto y, auto& delta_nabla_w, auto& delta_nabla_b) {

		size_t num_param_layers{ num_layers - 1 };

		for (size_t i = 0; i < num_param_layers; i++) {
			delta_nabla_b[i] = Matrix<float>::zeros(biases[i].rows, biases[i].cols);
			delta_nabla_w[i] = Matrix<float>::zeros(weights[i].rows, weights[i].cols);
		}

		auto activation = x;
		std::vector<Matrix<float>> activations{ x };
		std::vector<Matrix<float>> zs;

		for (size_t i = 0; i < biases.size(); i++) {
			const auto& b = biases[i];
			const auto& w = weights[i];

			auto z{ w * activation + b };
			zs.push_back(z);
			activation = act::sigmoid(z);
			activations.push_back(activation);
		}

		auto delta = cost_derivative(activations.back(), y) * act::sigmoid_prime(zs.back());
		delta_nabla_b.back() = delta;
		delta_nabla_w.back() = delta * activations[activations.size() - 2].transpose();

		for (size_t l = 2; l < num_layers; l++) {
			auto z = zs[zs.size() - l];
			auto sp = act::sigmoid_prime(z);
			delta = transpose(weights[weights.size() - l + 1]) * delta;
			delta = delta.hadamard(sp);

			size_t target = num_param_layers - l;
			delta_nabla_b[target] = delta;
			delta_nabla_w[target] = delta * transpose(activations[activations.size() - l - 1]);
		}
	}

	int evaluate(auto test_data) {
		std::vector<std::pair<size_t, int>> test_results;
		test_results.reserve(test_data.size());

		for (const auto& [x, y] : test_data) {
			auto output = feedforward(x);

			auto max_it = std::max_element(output.rix.begin(), output.rix.end());
			size_t argmax = std::distance(output.rix.begin(), max_it);

			test_results.push_back({ argmax, y });
		}

		int sum{ 0 };
		for (const auto& [pred, actual] : test_results) {
			if (pred == static_cast<size_t>(actual)) {
				sum++;
			}
		}
		return sum;
	}

	auto cost_derivative(auto output_activations, auto y) {
		return output_activations - y;
	}
};