#include <vector>	 // vector
#include <assert.h>	 // assert
#include <random>	 // shuffle, mt19937, random_device
#include <algorithm> // max_element, min
#include <fstream> 	 // ifstream, ofstream
#include <Spalten/Matrix.hpp>
#include "NN.hpp"
#include "activation_functions.hpp"
#include "utils.hpp"

Network::Network(std::vector<int> nw_sizes)
	: num_layers(nw_sizes.size()), 
	  sizes(std::move(nw_sizes)), 
	  num_param_layers(num_layers - 1), eta(0), epochs(0), test_data_provided(true)
{
	assert(sizes.size() > 1);

	// biases and weights are vectors of Matrix objects.
	// The ith Matrix represents the weights/biases of the (i+1)th layer.
	biases.reserve(num_param_layers);
	weights.reserve(num_param_layers);

	activations_buf.reserve(num_param_layers + 1);
	zs_buf.reserve(num_param_layers);
	nabla_w_buf.reserve(num_param_layers);
	nabla_b_buf.reserve(num_param_layers);


	// Matrix Initialization
	for (size_t i = 0; i < num_layers - 1; i++)
	{
		int x{sizes[i]};
		int y{sizes[i + 1]};
		// Left layer has x neurons, right layer has y neurons, so the weight matrix is y rows by x columns
		// This makes it multipliable with the left layer's output vector (x rows by 1 column)
		weights.emplace_back(mat_random_normal(y, x));
		nabla_w_buf.emplace_back(Matrix<float>::zeros(y, x));
	}
	for (size_t i = 1; i < num_layers; i++)
	{
		int y{sizes[i]};
		biases.emplace_back(mat_random_normal(y, 1));
		nabla_b_buf.emplace_back(Matrix<float>::zeros(y, 1));
	}
	
}

Network::Network(const std::string& model_path) : eta(0), epochs(0), test_data_provided(true) {
	std::ifstream model(model_path, std::ios::binary);

	if (!model.is_open()) throw std::runtime_error("Couldn't open model file.");

	// Number of layers
	model.read( reinterpret_cast<char*>(&num_layers), sizeof(size_t) );
	num_param_layers = num_layers - 1;

	// Neurons per layer
	sizes.resize(num_layers);
	model.read( reinterpret_cast<char*>(sizes.data()), num_layers * sizeof(int) );

	biases.reserve(num_param_layers);
	weights.reserve(num_param_layers);
	activations_buf.reserve(num_param_layers + 1);
	zs_buf.reserve(num_param_layers);
	nabla_w_buf.reserve(num_param_layers);
	nabla_b_buf.reserve(num_param_layers);

	// Matrix Initialization
	for (size_t i = 1; i < num_layers; i++)
	{
		int y{sizes[i]};
		biases.emplace_back(Matrix<float>::zeros(y, 1));
		nabla_b_buf.emplace_back(Matrix<float>::zeros(y, 1));
	}
	for (size_t i = 0; i < num_layers - 1; i++)
	{
		int x{sizes[i]};
		int y{sizes[i + 1]};
		// Left layer has x neurons, right layer has y neurons, so the weight matrix is y rows by x columns
		// This makes it multipliable with the left layer's output vector (x rows by 1 column)
		weights.emplace_back(Matrix<float>::zeros(y, x));
		nabla_w_buf.emplace_back(Matrix<float>::zeros(y, x));
	}

	// Weights
	for (auto& weight : weights) {
		model.read( reinterpret_cast<char*>(&weight.rows), sizeof(size_t));
		model.read( reinterpret_cast<char*>(&weight.cols), sizeof(size_t));
		model.read( reinterpret_cast<char*>(weight.rix.data()), weight.rows * weight.cols * sizeof(float));
	}

	// Biases
	for (auto& bias : biases) {
		model.read( reinterpret_cast<char*>(&bias.rows), sizeof(size_t));
		model.read( reinterpret_cast<char*>(&bias.cols), sizeof(size_t));
		model.read( reinterpret_cast<char*>(bias.rix.data()), bias.rows * bias.cols * sizeof(float));
	}
}

void Network::SGD(TrainingData training_data, int epochs,
				  int min_batch_size, float eta, const TestData& test_data)
{
	if (test_data.empty()) test_data_provided = false;
	this->eta = eta; 
	this->epochs = epochs;

	assert(min_batch_size > 0);
	assert(epochs > 0);
	assert(eta > 0.0F);
	assert(training_data.size() > 0);

	for (int i = 0; i < num_layers; i++) {
		activations_buf.emplace_back(Matrix<float>::zeros(sizes[i], min_batch_size));
		if (i == 0) continue;
		zs_buf.emplace_back(Matrix<float>::zeros(sizes[i], min_batch_size));
	}

	size_t n_test = test_data.size();	   // Number of testing pairs
	size_t n_train = training_data.size(); // Number of training pairs

	for (int j = 0; j < epochs; j++)
	{
		// Shuffling ensures that a lot of similar data isn't batched together (due to sorted datasets), 
		// 		which can lead to slower descent and overfitting.
		// It also ensures that the model sees a diverse set of examples in each epoch.
		std::shuffle(training_data.begin(), training_data.end(),
					 std::mt19937{std::random_device{}()});

		std::vector<TrainingData> mini_batches;
		mini_batches.reserve((n_train + min_batch_size) / min_batch_size);

		// Create mini-batches from the shuffled training data. The mini-batches span the entire training dataset.
		for (size_t k = 0; k < n_train; k += min_batch_size)
		{
			auto current_size = std::min(min_batch_size, static_cast<int>(n_train - k));
			mini_batches.emplace_back(
				std::vector<TrainingSample>(training_data.begin() + k,
											training_data.begin() + k + current_size));
		}

		for (const auto &mini_batch : mini_batches)
		{
			update_mini_batch(mini_batch, eta);
		}

		if (!test_data.empty())
			std::cout << std::format("Epoch {}: {} / {}", j, evaluate(test_data), n_test) << "\n";
		else
			std::cout << std::format("Epoch {} complete", j) << "\n";
	}
}

void Network::update_mini_batch(const std::vector<TrainingSample> &mini_batch, float eta)
{
	size_t m = mini_batch.size();
	size_t input_size = sizes.front();
	size_t output_size = sizes.back();
	Matrix<float> X(input_size, m);
	Matrix<float> Y(output_size, m);

	// Stacking all mini-batch input and output matrices into X and Y respectively. 
	// Each column of X corresponds to an input sample, and each column of Y corresponds to the corresponding output sample.
	// This allows for much faster matrix operations during backpropagation, as we can process the entire mini-batch in one go.
	for (size_t s = 0; s < m; s++)
	{
		const auto &[x, y] = mini_batch[s];
		for (size_t r = 0; r < input_size; r++)
			X(r, s) = x[r];
		for (size_t r = 0; r < output_size; r++)
			Y(r, s) = y[r];
	}

	activations_buf[0] = X;

	for (auto& nabla_w : nabla_w_buf) nabla_w.fill_zeros();
	for (auto& nabla_b : nabla_b_buf) nabla_b.fill_zeros();

	// Backprop computes the gradients of the cost function with respect to the weights and biases for the entire mini-batch.
	// The updates to nabla_w and nabla_b are computed in place,
	backprop(X, Y);

	// The scale factor does two things at once: 
	// 		- When training in mini-batches, the gradients calculuated must be averaged over the mini-batch size.
	// 		- The learning rate is applied to the averaged gradients to update the weights and biases.
	// Instead of dividing by the batch size in every iteration of the weight/bias update, 
	// 		we compute the scale factor once and multiply it with the gradients.
	float scale = eta / static_cast<float>(m);

	// The scaled gradients are subtracted from the current weights and biases to update them in the direction that minimizes the cost function.
	for (size_t i = 0; i < num_param_layers; i++)
	{
		weights[i] -= nabla_w_buf[i] * scale;
		biases[i] -= nabla_b_buf[i] * scale;
	}
}

void Network::backprop(const Matrix<float> &X, const Matrix<float> &actual_result)
{
	Matrix<float> activation(X); // Number of neurons in input layer = 10 * Minibatch size = 10x10

	// Feedforward pass: Compute the activations and weighted inputs (z) for each layer.
	for (size_t i = 0; i < biases.size(); i++)
	{
		const Matrix<float> &b = biases[i];			// [neurons x 1]
		const Matrix<float> &w = weights[i];		// [neurons x prev_neurons]
		Matrix<float> z = (w * activation) + b; 	// [neurons x m] 128x10 times 10x10 = 128x10 + 128x1 = 128x10
		zs_buf[i] = z;
		activation = act::sigmoid(z);
		activations_buf[i + 1] = activation;
	}

	// Backward pass: Compute the gradients of the cost function with respect to weights and biases using the chain rule.

	// Error is the derivative of the cost function wrt. the activations of the output layer,
	// 		multiplied element-wise by the derivative of the activation function wrt. the weighted input (z) of the output layer.
	Matrix<float> inter(zs_buf.back());
	act::sigmoid_prime(zs_buf.back(), inter);
	Matrix<float> error = cost_derivative(activations_buf.back(), actual_result).hadamard(inter);
							
	nabla_b_buf.back() = row_sum(error);
	nabla_w_buf.back() = error * transpose( activations_buf[activations_buf.size() - 2] );

	for (size_t l = 2; l < num_layers; l++)
	{
		size_t target = num_param_layers - l;
		const auto &z = zs_buf[target];
		Matrix<float> d_act(z.rows, z.cols);
		act::sigmoid_prime(z, d_act);
		error = transpose(weights[target + 1]) * error;
		error = error.hadamard(d_act);
		nabla_b_buf[target] = row_sum(error);
		nabla_w_buf[target] = error * transpose(activations_buf[target]);
	}
}

Matrix<float> Network::cost_derivative(const Matrix<float> &output_activations,
									   const Matrix<float> &actual_result) {

	return output_activations - actual_result;
}

Matrix<float> Network::feedforward(Matrix<float> a) const
{
	for (size_t i = 0; i < num_param_layers; i++)
	{
		a = act::sigmoid(weights[i] * a + biases[i]);
	}
	return a;
}

Matrix<float> Network::row_sum(const Matrix<float> &mat)
{
	Matrix<float> result(mat.rows, 1);
	for (size_t r = 0; r < mat.rows; r++)
	{
		float s = 0.0F;
		for (size_t c = 0; c < mat.cols; c++)
			s += mat(r, c);
		result(r, 0) = s;
	}
	return result;
}

int Network::evaluate(const TestData& test_data) const
{
	std::vector<std::pair<size_t, int>> test_results;
	test_results.reserve(test_data.size());

	for (const auto &[x, y] : test_data)
	{
		auto output = feedforward(x);

		auto max_it = std::max_element(output.rix.begin(), output.rix.end());
		size_t argmax = std::distance(output.rix.begin(), max_it);

		test_results.push_back({argmax, y});
	}

	int sum{0};
	for (const auto &[pred, actual] : test_results)
	{
		if (pred == static_cast<size_t>(actual))
			sum++;
	}
	return sum;
}

// Use forward dashes!
void Network::export_model(std::string model_directory, std::string dataset_name) {
	std::string model_full_path(std::move(dataset_name));

	model_full_path += "_";
	for (const auto& size : sizes) {
		model_full_path += std::to_string(size) + "-";
	}
	model_full_path.back() = '_';
	model_full_path += "ep" + std::to_string(epochs) + "_lr" + float_to_string(eta);
	// TODO: Support for Loss in the filename

	if (model_directory.back() != '/') model_directory.append("/");
	model_full_path = model_directory + model_full_path + ".bin";

	std::ofstream model(model_full_path, std::ios::binary);

	if (!model.is_open()) throw std::runtime_error("Couldn't open model output file.");

	// Number of layers
	model.write( reinterpret_cast<char*>(&num_layers), sizeof(size_t) );

	// Neurons per layer
	model.write( reinterpret_cast<char*>(sizes.data()), num_layers * sizeof(int) );

	// Weights
	for (auto& weight : weights) {
		model.write( reinterpret_cast<char*>(&weight.rows), sizeof(size_t));
		model.write( reinterpret_cast<char*>(&weight.cols), sizeof(size_t));
		model.write( reinterpret_cast<char*>(weight.rix.data()), weight.rows * weight.cols * sizeof(float));
	}

	// Biases
	for (auto& bias : biases) {
		model.write( reinterpret_cast<char*>(&bias.rows), sizeof(size_t));
		model.write( reinterpret_cast<char*>(&bias.cols), sizeof(size_t));
		model.write( reinterpret_cast<char*>(bias.rix.data()), bias.rows * bias.cols * sizeof(float));
	}
}