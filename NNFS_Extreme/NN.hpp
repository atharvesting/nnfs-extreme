#pragma once
#include <vector>

#include <Spalten/Matrix.hpp>

using TrainingSample	= std::pair<Matrix<float>, Matrix<float>>;
using TrainingData		= std::vector<TrainingSample>;
using TestSample		= std::pair<Matrix<float>, int>;
using TestData			= std::vector<TestSample>;

class Network {
public:
	int num_layers;
	std::vector<int> sizes;
	std::vector<Matrix<float>> biases;
	std::vector<Matrix<float>> weights;

	Network(std::vector<int> sizes);
	
	Matrix<float> feedforward(Matrix<float> a) const;

	void update_mini_batch(const std::vector<TrainingSample>& mini_batch, float eta);

	void SGD(TrainingData training_data, int epochs,
					  int min_batch_size, float learning_rate,
					  TestData test_data);

	void backprop(const Matrix<float>& X, const Matrix<float>& Y,
		std::vector<Matrix<float>>& nabla_w,
		std::vector<Matrix<float>>& nabla_b);

	int evaluate(TestData test_data);

	Matrix<float> cost_derivative(const Matrix<float>& output_activations,
								  const Matrix<float>& y) const;

private:
	Matrix<float> row_sum(const Matrix<float>& m) const;
};
