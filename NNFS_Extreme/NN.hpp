#pragma once
#include <vector>
#include <Spalten/Matrix.hpp>

// Traditional MNIST: TrainingSample contains a pair of InputMatrix(784 x 1) and OutputMatrix(10, 1)
using TrainingSample	= std::pair<Matrix<float>, Matrix<float>>;
// Traditional MNIST: TrainingData simply contains a vector of TrainingSample's
using TrainingData		= std::vector<TrainingSample>;
// Traditional MNIST: TestSample contains a pair of InputMatrix(784 x 1) and DigitLabel(int)
using TestSample		= std::pair<Matrix<float>, int>;
// Traditional MNIST: TestData simply contains a vector of TestSample's
using TestData			= std::vector<TestSample>;

class Network {
public:
	size_t num_layers;
	size_t num_param_layers;
	std::vector<int> sizes;
	std::vector<Matrix<float>> biases;
	std::vector<Matrix<float>> weights;
	bool test_data_provided;
	int epochs;
	int eta;
	
	// '_buf' postfix indicating that the container is reused to avoid allocation overhead.
	std::vector<Matrix<float>> activations_buf;
	std::vector<Matrix<float>> zs_buf;
	
	/// @param nabla_w_buf Gradients of the cost function with respect to the weights. Zeroed out after every mini-batch update.
	std::vector<Matrix<float>> nabla_w_buf;
	
	/// @param nabla_b_buf Gradients of the cost function with respect to the biases. Zeroed out after every mini-batch update.
	std::vector<Matrix<float>> nabla_b_buf;

	/// @brief Constructs a neural network with the specified layer sizes.
	/// @param nw_sizes A vector containing the number of neurons in each layer (including input and output layers).
	explicit Network(std::vector<int> sizes);

	/// @brief Constructs a neural network using the file containing network data.
	/// @param model_path Path to the network data.
	explicit Network(const std::string& model_path);

	/// @brief Feeds forward the input through the network.
	/// @param a Input matrix.
	/// @return Output matrix.
	Matrix<float> feedforward(Matrix<float> a) const;

	/// @brief Stochastic Gradient Descent (SGD) algorithm for training the neural network.
	/// @param training_data A vector of TrainingSample pairs (input matrix, output matrix).
	/// @param epochs Number of epochs to train for.
	/// @param min_batch_size Size of each mini-batch.
	/// @param eta Learning rate.
	/// @param test_data A vector of TestSample pairs (input matrix, digit label).
	/// @return A pair containing the final weights and biases after training.
	void SGD(TrainingData training_data, int epochs, int min_batch_size, float eta, const TestData& test_data);

	/// @brief Export the model parameters as a custom binary. Imported using the file constructor.
	/// @param model_path The path+filename as the export location.
	void export_model(std::string model_path, std::string dataset_name);

private:
	/// @brief Updates the network's weights and biases using a mini-batch of training data.
	/// @param mini_batch A vector of TrainingSample pairs (input matrix, output matrix).
	/// @param eta Learning rate.
	void update_mini_batch(const std::vector<TrainingSample>& mini_batch, float eta);

	/// @brief Computes the gradients of the cost function with respect to the weights and biases.
	/// @param X Input matrix.
	/// @param Y True labels.
	void backprop(const Matrix<float>& X, const Matrix<float>& actual_result);

	/// @brief Evaluates the network on the test data.
	/// @param test_data A vector of TestSample pairs (input matrix, digit label).
	/// @return The number of correctly classified samples.
	int evaluate(const TestData& test_data) const;

	/// @brief Computes the derivative of the cost function with respect to the output activations.
	/// @param output_activations The output activations of the network.
	/// @param actual_result The true labels.
	/// @return The derivative of the cost function.
	static Matrix<float> cost_derivative(const Matrix<float>& output_activations, const Matrix<float>& actual_result);

	/// @brief Computes the sum of elements in each row of a matrix.
	/// @param mat The input matrix.
	/// @return A matrix containing the sum of elements in each row.
	static Matrix<float> row_sum(const Matrix<float>& m);
	
};
