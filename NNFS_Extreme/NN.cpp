#include <vector>	 // vector
#include <assert.h>	 // assert
#include <random>	 // shuffle, mt19937, random_device
#include <algorithm> // max_element, min, 
#include <Spalten/Matrix.hpp>
#include "NN.hpp"
#include "activation_functions.hpp"

/// @brief Constructs a neural network with the specified layer sizes.
/// @param nw_sizes A vector containing the number of neurons in each layer (including input and output layers).
Network::Network(std::vector<int> nw_sizes)
	: num_layers(nw_sizes.size()), sizes(std::move(nw_sizes)), num_param_layers(num_layers - 1)
{
	assert(sizes.size() > 1);

	// biases and weights are vectors of Matrix objects.
	// The ith Matrix represents the weights/biases of the (i+1)th layer.
	biases.reserve(num_layers - 1);
	weights.reserve(num_layers - 1);

	// Matrix Initialization
	for (size_t i = 1; i < num_layers; i++)
	{
		int y{sizes[i]};
		biases.emplace_back(mat_random_normal(y, 1));
	}
	for (size_t i = 0; i < num_layers - 1; i++)
	{
		int x{sizes[i]};
		int y{sizes[i + 1]};
		// Left layer has x neurons, right layer has y neurons, so the weight matrix is y rows by x columns
		// This makes it multipliable with the left layer's output vector (x rows by 1 column)
		weights.emplace_back(mat_random_normal(y, x));
	}
}

/// @brief Stochastic Gradient Descent (SGD) algorithm for training the neural network.
/// @param training_data A vector of TrainingSample pairs (input matrix, output matrix).
/// @param epochs Number of epochs to train for.
/// @param min_batch_size Size of each mini-batch.
/// @param eta Learning rate.
/// @param test_data A vector of TestSample pairs (input matrix, digit label).
/// @return A pair containing the final weights and biases after training.
std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>> Network::SGD(TrainingData training_data, int epochs,
				  int min_batch_size, float eta, const TestData& test_data)
{
	assert(min_batch_size > 0);
	assert(epochs > 0);
	assert(eta > 0.0F);
	assert(training_data.size() > 0);
	assert(test_data.size() > 0);

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
	return std::make_pair(weights, biases);
}

/// @brief Updates the network's weights and biases using a mini-batch of training data.
/// @param mini_batch A vector of TrainingSample pairs (input matrix, output matrix).
/// @param eta Learning rate.
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

	std::vector<Matrix<float>> nabla_b;
	std::vector<Matrix<float>> nabla_w;
	nabla_b.reserve(num_param_layers);
	nabla_w.reserve(num_param_layers);

	for (size_t i = 0; i < num_param_layers; i++)
	{
		nabla_b.emplace_back(Matrix<float>::zeros(biases[i].rows, biases[i].cols));
		nabla_w.emplace_back(Matrix<float>::zeros(weights[i].rows, weights[i].cols));
	}
	// Backprop computes the gradients of the cost function with respect to the weights and biases for the entire mini-batch.
	// The updates to nabla_w and nabla_b are computed in place,
	backprop(X, Y, nabla_w, nabla_b);

	// The scale factor does two things at once: 
	// 		- When training in mini-batches, the gradients calculuated must be averaged over the mini-batch size.
	// 		- The learning rate is applied to the averaged gradients to update the weights and biases.
	// Instead of dividing by the batch size in every iteration of the weight/bias update, 
	// 		we compute the scale factor once and multiply it with the gradients.
	float scale = eta / static_cast<float>(m);

	// The scaled gradients are subtracted from the current weights and biases to update them in the direction that minimizes the cost function.
	for (size_t i = 0; i < num_param_layers; i++)
	{
		weights[i] -= nabla_w[i] * scale;
		biases[i] -= nabla_b[i] * scale;
	}
}

/// @brief Computes the gradients of the cost function with respect to the weights and biases.
/// @param X Input matrix.
/// @param Y True labels.
/// @param nabla_w Gradients of the cost function with respect to the weights. Zeroed out after every mini-batch update.
/// @param nabla_b Gradients of the cost function with respect to the biases. Zeroed out after every mini-batch update.
void Network::backprop(const Matrix<float> &X, const Matrix<float> &actual_result,
					   std::vector<Matrix<float>> &nabla_w,
					   std::vector<Matrix<float>> &nabla_b)
{
	Matrix<float> activation = X;
	std::vector<Matrix<float>> activations{X};
	std::vector<Matrix<float>> zs;

	// Feedforward pass: Compute the activations and weighted inputs (z) for each layer.
	for (size_t i = 0; i < biases.size(); i++)
	{
		const Matrix<float> &b = biases[i];		// [neurons x 1]
		const Matrix<float> &w = weights[i];	// [neurons x prev_neurons]
		Matrix<float> z = w * activation + b; 	// [neurons x m]
		zs.push_back(z);
		activation = act::sigmoid(z);
		activations.push_back(activation);
	}

	// Backward pass: Compute the gradients of the cost function with respect to weights and biases using the chain rule.

	// Error is the derivative of the cost function wrt. the activations of the output layer,
	// 		multiplied element-wise by the derivative of the activation function wrt. the weighted input (z) of the output layer.
	Matrix<float> error = cost_derivative(activations.back(), actual_result)
							.hadamard( act::sigmoid_prime(zs.back()) );
	nabla_b.back() = row_sum(error);
	nabla_w.back() = error * transpose( activations[activations.size() - 2] );

	for (size_t l = 2; l < num_layers; l++)
	{
		size_t target = num_param_layers - l;
		const auto &z = zs[target];
		auto sp = act::sigmoid_prime(z);
		error = transpose(weights[target + 1]) * error;
		error = error.hadamard(sp);
		nabla_b[target] = row_sum(error);
		nabla_w[target] = error * transpose(activations[target]);
	}
}

/// @brief Computes the derivative of the cost function with respect to the output activations.
/// @param output_activations The output activations of the network.
/// @param actual_result The true labels.
/// @return The derivative of the cost function.
Matrix<float> Network::cost_derivative(const Matrix<float> &output_activations,
									   const Matrix<float> &actual_result) {

	return output_activations - actual_result;
}

/// @brief Feeds forward the input through the network.
/// @param a Input matrix.
/// @return Output matrix.
Matrix<float> Network::feedforward(Matrix<float> a) const
{
	for (size_t i = 0; i < num_param_layers; i++)
	{
		a = act::sigmoid(weights[i] * a + biases[i]);
	}
	return a;
}

/// @brief Computes the sum of elements in each row of a matrix.
/// @param mat The input matrix.
/// @return A matrix containing the sum of elements in each row.
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

/// @brief Evaluates the network on the test data.
/// @param test_data A vector of TestSample pairs (input matrix, digit label).
/// @return The number of correctly classified samples.
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