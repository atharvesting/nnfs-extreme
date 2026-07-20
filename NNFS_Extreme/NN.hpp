#pragma once
#include <vector>
#include <Spalten/Matrix.hpp>

// MNIST: TrainingSample contains a pair of InputMatrix(784 x 1) and OutputMatrix(10, 1)
using TrainingSample	= std::pair<Matrix<float>, Matrix<float>>;
// MNIST: TrainingData simply contains a vector of TrainingSample's
using TrainingData		= std::vector<TrainingSample>;
// MNIST: TestSample contains a pair of InputMatrix(784 x 1) and DigitLabel(int)
using TestSample		= std::pair<Matrix<float>, int>;
// MNIST: TestData simply contains a vector of TestSample's
using TestData			= std::vector<TestSample>;

class Network {
public:
	size_t num_layers;
	size_t num_param_layers;
	std::vector<int> sizes;
	std::vector<Matrix<float>> biases;
	std::vector<Matrix<float>> weights;

	Network(std::vector<int> sizes);
	Matrix<float> feedforward(Matrix<float> a) const;
	std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>> SGD(TrainingData training_data, int epochs, int min_batch_size, float eta, const TestData& test_data);

private:
	void update_mini_batch(const std::vector<TrainingSample>& mini_batch, float eta);
	void backprop(const Matrix<float>& X, const Matrix<float>& actual_result, std::vector<Matrix<float>>& nabla_w, std::vector<Matrix<float>>& nabla_b);
	int evaluate(const TestData& test_data) const;
	static Matrix<float> cost_derivative(const Matrix<float>& output_activations, const Matrix<float>& actual_result);
	static Matrix<float> row_sum(const Matrix<float>& m);
	
};
