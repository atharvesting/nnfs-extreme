#include "NN.hpp"
#include "data_loaders.hpp"
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <source_location>
#include <filesystem>

void require(bool ok, std::source_location location = std::source_location::current()) {
    if (!ok) throw std::runtime_error("Benchmark self-check failed at line " + std::to_string(location.line()));
}
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}

int reference_correct(const Network& net, const TestData& data) {
    int correct = 0;
    for (const auto& [x, label] : data) {
        auto output = net.feedforward(x);
        auto prediction = std::max_element(output.rix.begin(), output.rix.end()) - output.rix.begin();
        correct += prediction == label;
    }
    return correct;
}

void check_evaluation() {
    auto train = std::make_shared<const TrainingData>();
    auto data = std::make_shared<TestData>(MNIST_loader::load_test_data(
        "data/mnist_test_images.bin", "data/mnist_test_labels.bin", 137));
    for (auto activation : {act::Type::Sigmoid, act::Type::Relu, act::Type::Tanh, act::Type::LeakyRelu}) {
        Config config(1, 32, 0.1F, {784, 30, 10});
        config.activation = activation;
        Network net(config, train, data);
        // Check every prediction, including a nine-image final batch.
        for (size_t start = 0; start < data->size(); start += 128) {
            size_t count = std::min(size_t{128}, data->size() - start);
            Matrix<float> x(784, count);
            for (size_t r = 0; r < 784; ++r)
                for (size_t c = 0; c < count; ++c) x(r, c) = (*data)[start + c].first[r];
            auto batch = net.feedforward(std::move(x));
            for (size_t c = 0; c < count; ++c) {
                auto single = net.feedforward((*data)[start + c].first);
                size_t prediction = 0;
                for (size_t r = 1; r < batch.rows; ++r)
                    if (batch(r, c) > batch(prediction, c)) prediction = r;
                require(prediction == static_cast<size_t>(std::max_element(single.rix.begin(), single.rix.end()) - single.rix.begin()));
            }
        }
        net.SGD(false); // Empty training data leaves weights unchanged; exercises actual evaluate().
        require(std::lround(net.observe.final_accuracy * data->size() / 100.0F) == reference_correct(net, *data));
        // ReLU zero outputs are bit-exact ties, unlike fast-math sigmoid approximations.
        net.config.activation = act::Type::Relu;
        for (auto& matrix : net.weights) matrix.fill_zeros();
        for (auto& matrix : net.biases) matrix.fill_zeros();
        net.SGD(false);
        require(std::lround(net.observe.final_accuracy * data->size() / 100.0F) == reference_correct(net, *data));
    }
    auto full = std::make_shared<const TestData>(MNIST_loader::load_test_data(
        "data/mnist_test_images.bin", "data/mnist_test_labels.bin", 10000));
    Network large(Config(1, 32, 0.1F, {784, 512, 512, 10}), train, full);
    Timer timer;
    int expected = reference_correct(large, *full);
    double baseline = timer.elapsed();
    large.SGD(false);
    require(std::abs(large.observe.final_accuracy - expected / 100.0F) < 1e-4F);
    std::cout << "Largest topology, 10000 images: individual=" << baseline
              << "s, batched=" << large.observe.total_testing_time
              << "s, speedup=" << baseline / large.observe.total_testing_time << "x\n";
}

int main() {
    check_evaluation();
    Odometer odo({2, 3, 2});
    std::set<std::vector<int>> states;
    do { require(states.insert(odo.get_state()).second); } while (odo.next());
    require(states.size() == 12);
    Config config(2, 5, 0.1F, {784, 3, 10});
    config.samples_per_worker = 2;
    config.thread_count = 3;
    auto train = std::make_shared<TrainingData>();
    auto test = std::make_shared<TestData>();
    for (int i = 0; i < 7; ++i) {
        Matrix<float> x(784, 1, 0), y(10, 1, 0);
        x[i] = 0.5F;
        y[i % 10] = 1;
        train->emplace_back(x, y);
        test->emplace_back(x, i % 10);
    }
    for (auto activation : {act::Type::Sigmoid, act::Type::Relu, act::Type::Tanh, act::Type::LeakyRelu}) {
        config.activation = activation;
        config.thread_state = Threading::Single;
        Network single(config, train, test), repeat(config, train, test);
        config.thread_state = Threading::Multi;
        Network multi(config, train, test);
        single.SGD(false); repeat.SGD(false); multi.SGD(false);
        require(single.observe.accuracy_progression.size() == 2);
        for (size_t l = 0; l < single.weights.size(); ++l) {
            for (size_t i = 0; i < single.weights[l].rix.size(); ++i) {
                require(single.weights[l].rix[i] == repeat.weights[l].rix[i]);
                require(std::abs(single.weights[l].rix[i] - multi.weights[l].rix[i]) < 1e-5F);
            }
            for (size_t i = 0; i < single.biases[l].rix.size(); ++i)
                require(std::abs(single.biases[l].rix[i] - multi.biases[l].rix[i]) < 1e-5F);
        }
        require(single.observe.accuracy_progression == multi.observe.accuracy_progression);
    }
    config.acc_threshold = 50;
    Observability observe(config);
    observe.update(2, 1, 40); observe.update(3, 1, 50); observe.process();
    require(observe.epochs_to_accuracy_threshold == 2 && observe.time_to_accuracy_threshold == 7);
    require(std::abs(observe.total_time - 7) < 1e-5F && std::abs(observe.avg_training_time - 2.5F) < 1e-5F);
    config.acc_threshold = 90;
    Observability unreached(config);
    unreached.update(2, 1, 40); unreached.process();
    require(unreached.epochs_to_accuracy_threshold == 0 && unreached.time_to_accuracy_threshold == 0);
    // Infinite final biases must not silently produce a digit prediction.
    config.activation = act::Type::Relu;
    config.thread_state = Threading::Single;
    Network divergent(config, train, test);
    for (auto& value : divergent.biases.back().rix) value = std::numeric_limits<float>::infinity();
    rejects([&] { divergent.SGD(false); });
    std::ofstream("empty.bin");
    rejects([] { MNIST_loader::load_training_data("empty.bin", "empty.bin", 1, false); });
    rejects([] { MNIST_loader::load_test_data("empty.bin", "empty.bin", 1); });
    BenchConfig invalid;
    invalid.activation.clear(); invalid.param_options_count.back() = 0;
    rejects([&] { Benchmark(invalid).run(); });
    invalid = BenchConfig(); invalid.eta = {std::numeric_limits<float>::infinity()};
    rejects([&] { Benchmark(invalid).run(); });
    // Real MNIST, two short configurations: epoch propagation and JSONL persistence.
    BenchConfig smoke(1, {256}, {0.1F}, {{784, 2, 10}},
        {Threading::Single, Threading::Multi}, {42},
        {init::Type::XavierNormal}, {act::Type::Sigmoid}, 3, 32);
    std::ofstream("results.jsonl") << "existing results\n";
    Benchmark(smoke).run();
    std::filesystem::path output;
    for (const auto& entry : std::filesystem::directory_iterator("."))
        if (entry.path().filename().string().starts_with("results_") && entry.path().extension() == ".jsonl") {
            require(output.empty());
            output = entry.path();
        }
    require(!output.empty());
    std::ifstream file(output);
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        auto row = json::parse(line);
        require(row["evaluation_method"] == "batched_128_v1");
        require(row["config"]["epochs"] == 1 && row["status"] == "completed");
        require(row["run results"]["accuracy_progression"].size() == 1);
        require(row["training_samples"] == 50000 && row["test_samples"] == 10000);
        require(row["config_no"] == ++count);
    }
    require(count == 2);
    std::ifstream preserved("results.jsonl");
    std::getline(preserved, line);
    require(line == "existing results");
    std::cout << "Benchmark checks passed.\n";
}
