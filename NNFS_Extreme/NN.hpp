#pragma once
#include <vector>
#include <Spalten/Matrix.hpp>
#include "activation_functions.hpp"
#include <format>
#include <iostream>

using Sample = std::pair<Matrix<float>, Matrix<float>>;
using TrainingData = std::vector<Sample>;
using TestSample = std::pair<Matrix<float>, int>;
using TestData = std::vector<TestSample>;

class Network {
public:
    explicit Network(const std::vector<int>& sizes);

    // concrete-type interface so implementations can live in NN.cpp
    Matrix<float> feedforward(Matrix<float> a) const;
    void update_mini_batch(const std::vector<Sample>& mini_batch, float eta);
    void SGD(TrainingData& training_data, int epochs, int min_batch_size, float learning_rate, const TestData& test_data);

    // backprop works on concrete matrix types
    void backprop(const Matrix<float>& x, const Matrix<float>& y,
        std::vector<Matrix<float>>& delta_nabla_w,
        std::vector<Matrix<float>>& delta_nabla_b);

    int evaluate(const TestData& test_data) const;

private:
    int num_layers;
    std::vector<int> sizes;
    std::vector<Matrix<float>> biases;
    std::vector<Matrix<float>> weights;

    Matrix<float> cost_derivative(const Matrix<float>& output_activations, const Matrix<float>& y) const;
};