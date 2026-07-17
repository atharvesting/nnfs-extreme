#include <vector> // vector
#include <assert.h> // assert
#include <random> // shuffle
#include <algorithm> // max_element, min

#include <Spalten/Matrix.hpp>
#include "NN.hpp"
#include "activation_functions.hpp"

Network::Network(std::vector<int> nw_sizes)
	: num_layers(nw_sizes.size()), sizes(std::move(nw_sizes)) {

	assert(sizes.size() > 1);

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

Matrix<float> Network::feedforward(Matrix<float> a) const {
	for (size_t i = 0; i < biases.size(); i++) {
		a = act::sigmoid(weights[i] * a + biases[i]);
	}
	return a;
}

void Network::update_mini_batch(const std::vector<TrainingSample>& mini_batch, float eta) {

	size_t num_param_layers = weights.size();
	size_t m = mini_batch.size();

	Matrix<float> X(sizes[0], m);
	Matrix<float> Y(sizes.back(), m);

	for (size_t s = 0; s < m; s++) {
		const auto& [x, y] = mini_batch[s];
		for (size_t r = 0; r < static_cast<size_t>(sizes[0]); r++)
			X(r, s) = x[r];
		for (size_t r = 0; r < static_cast<size_t>(sizes.back()); r++)
			Y(r, s) = y[r];
	}

	std::vector<Matrix<float>> nabla_b;
	std::vector<Matrix<float>> nabla_w;
	nabla_b.reserve(num_param_layers);
	nabla_w.reserve(num_param_layers);

	for (size_t i = 0; i < num_param_layers; i++) {
		nabla_b.emplace_back(Matrix<float>::zeros(biases[i].rows, biases[i].cols));
		nabla_w.emplace_back(Matrix<float>::zeros(weights[i].rows, weights[i].cols));
	}

	backprop(X, Y, nabla_w, nabla_b);

	const float scale = eta / static_cast<float>(m);

	for (size_t i = 0; i < num_param_layers; i++) {
		for (size_t j = 0; j < weights[i].size(); j++)
			weights[i][j] -= scale * nabla_w[i][j];
		for (size_t j = 0; j < biases[i].size(); j++)
			biases[i][j] -= scale * nabla_b[i][j];
	}
}

void Network::SGD(TrainingData training_data, int epochs,
				  int min_batch_size, float eta, TestData test_data) {

	size_t n_test = test_data.size();
	int n = training_data.size();

	for (int j = 0; j < epochs; j++) {
		std::shuffle(training_data.begin(), training_data.end(),
					 std::mt19937{ std::random_device{}() });

		std::vector<TrainingData> mini_batches;
		mini_batches.reserve((training_data.size() + min_batch_size) / min_batch_size);

		for (size_t k = 0; k < n; k += min_batch_size) {
			auto current_size = std::min(min_batch_size, static_cast<int>(n - k));
			mini_batches.emplace_back(
				std::vector<TrainingSample>(training_data.begin() + k,
											training_data.begin() + k + current_size)
			);
		}

		for (const auto& mini_batch : mini_batches) {
			update_mini_batch(mini_batch, eta);
		}
		std::cout << std::format("Epoch {}: {} / {}", j, evaluate(test_data), n_test) << std::endl;
	}
}

Matrix<float> Network::row_sum(const Matrix<float>& mat) const {
	Matrix<float> result(mat.rows, 1);
	for (size_t r = 0; r < mat.rows; r++) {
		float s = 0.0f;
		for (size_t c = 0; c < mat.cols; c++)
			s += mat(r, c);
		result(r, 0) = s;
	}
	return result;
}

void Network::backprop(const Matrix<float>& X, const Matrix<float>& Y,
					   std::vector<Matrix<float>>& nabla_w,
					   std::vector<Matrix<float>>& nabla_b) {

	size_t num_param_layers = num_layers - 1;

	Matrix<float> activation = X;
	std::vector<Matrix<float>> activations{ X };
	std::vector<Matrix<float>> zs;

	for (size_t i = 0; i < biases.size(); i++) {
		const Matrix<float>& b = biases[i];  // [neurons x 1]
		const Matrix<float>& w = weights[i]; // [neurons x prev_neurons]

		Matrix<float> z = w * activation + b; // [neurons x m], bias broadcasts
		zs.push_back(z);
		activation = act::sigmoid(z);
		activations.push_back(activation);
	}

	auto delta = cost_derivative(activations.back(), Y).hadamard(act::sigmoid_prime(zs.back()));
	nabla_b.back() = row_sum(delta);
	nabla_w.back() = delta * transpose(activations[activations.size() - 2]);

	for (size_t l = 2; l < num_layers; l++) {
		size_t target = num_param_layers - l;
		const auto& z  = zs[target];
		auto sp  = act::sigmoid_prime(z);
		delta = transpose(weights[target + 1]) * delta;
		delta = delta.hadamard(sp);
		nabla_b[target] = row_sum(delta);
		nabla_w[target] = delta * transpose(activations[target]);
	}
}

int Network::evaluate(TestData test_data) {
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
		if (pred == static_cast<size_t>(actual))
			sum++;
	}
	return sum;
}

Matrix<float> Network::cost_derivative(const Matrix<float>& output_activations, 
					 const Matrix<float>& y) const {

	return output_activations - y;
}