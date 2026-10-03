#include "examples.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NN.hpp"
#include "data_loaders.hpp"

void examples::standard_mnist()
{
    Config config(30, 64, 0.5F, {784, 512, 512, 10}, Threading::Multi, 42);
    const std::string thread_state = config.thread_state == Threading::Single ? "Single" : "Multi";

    std::cout << "Running standard MNIST example!\n"
              << "Epochs: " << config.epochs
              << ", Mini-batch size: " << config.mini_batch_size
              << ", Learning rate: " << config.eta
              << ", Threading: " << thread_state << '\n';

    Network(config).SGD();
}

void examples::inverted_mnist()
{
    Config config(20, 64, 15.0F, {10, 30, 784});
    std::cout << "Running inverted MNIST example!\n"
              << "Epochs: " << config.epochs
              << ", Mini-batch size: " << config.mini_batch_size
              << ", Learning rate: " << config.eta << '\n';

    Network nn(config);
    nn.training_data = std::make_shared<const TrainingData>(
        MNIST_loader::load_training_data(config.train_images, config.train_labels, 50000, true));
    nn.test_data = std::make_shared<const TestData>();
    nn.SGD();

    nn.export_model("data/output/", "mnist_inverted");

    const Matrix<float> output = nn.feedforward(Matrix<float>(10, 1, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    std::ofstream pixel_data("data/output/pixel_data.bin", std::ios::binary);
    if (!pixel_data.is_open())
        throw std::runtime_error("Could not open pixel_data.bin for writing.");

    pixel_data.write(reinterpret_cast<const char*>(output.rix.data()), output.rows * output.cols * sizeof(float));
}

void examples::model_export_n_import()
{
    Config config(30, 32, 9.0F, {10, 30, 784});
    Network nn(config);
    nn.training_data = std::make_shared<const TrainingData>(
        MNIST_loader::load_training_data(config.train_images, config.train_labels, 50000, true));
    nn.test_data = std::make_shared<const TestData>();
    nn.SGD();

    const auto model_path = nn.export_model("data/output/", "mnist_inverted");
    Network new_nn(model_path, config);

    const Matrix<float> output = new_nn.feedforward(Matrix<float>(10, 1, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}));
    std::ofstream pixel_data("data/output/pixel_data.bin", std::ios::binary);
    if (!pixel_data.is_open())
        throw std::runtime_error("Could not open pixel_data.bin for writing.");

    pixel_data.write(reinterpret_cast<const char*>(output.rix.data()), output.rows * output.cols * sizeof(float));
}

void examples::import_model()
{
    Config config(20, 64, 15.0F, {10, 30, 784});
    Network nn("data/output/mnist_inverted_10-30-784_ep20_lr15p000.bin", config);

    const Matrix<float> output = nn.feedforward(Matrix<float>(10, 1, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}));
    std::ofstream pixel_data("data/output/pixel_data_8.bin", std::ios::binary);
    if (!pixel_data.is_open())
        throw std::runtime_error("Could not open pixel_data_8.bin for writing.");

    pixel_data.write(reinterpret_cast<const char*>(output.rix.data()), output.rows * output.cols * sizeof(float));
}
