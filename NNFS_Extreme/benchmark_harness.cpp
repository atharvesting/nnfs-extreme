#include "benchmark_harness.hpp"
#include "NN.hpp"
#include "data_loaders.hpp"
#include "utils.hpp"
#include <algorithm> // max
#include <cstdlib>
#include <fstream>
#include <numeric>   // accumulate
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <sstream>
#include <limits>
#include <bit>
#include <cstdint>

Config::Config(int epochs, int mini_batch_size, float eta,
               std::vector<int> topology, Threading thread_state,
               int random_seed, init::Type initializer, act::Type activation,
               int thread_count, std::string train_images,
               std::string test_images, std::string train_labels,
               std::string test_labels, float acc_threshold, int samples_per_worker
               )
    : epochs(epochs), mini_batch_size(mini_batch_size), eta(eta),
      topology(topology), thread_state(thread_state),
      thread_count(thread_count), samples_per_worker(samples_per_worker), train_images(train_images),
      test_images(test_images), train_labels(train_labels),
      test_labels(test_labels), acc_threshold(acc_threshold),
      random_seed(random_seed), initializer(initializer), activation(activation) {}

json Config::to_json_object() {
    return json{
        {"epochs", epochs},
        {"mini_batch_size", mini_batch_size},
        {"eta", eta},
        {"topology", topology},
        {"thread_state", thread_state},
        {"thread_count", thread_count},
        {"samples_per_worker", samples_per_worker},
        {"random_seed", random_seed},
        {"initializer", init::name(initializer)},
        {"activation", act::name(activation)},
        {"acc_threshold", acc_threshold}
    };
}

BenchConfig::BenchConfig(
    size_t                           epochs,
    std::vector<int>        mini_batch_size,
    std::vector<float>                  eta,
    std::vector<std::vector<int>>  topology,
    std::vector<Threading>     thread_state,
    std::vector<int>            random_seed,
    std::vector<init::Type>     initializer,
    std::vector<act::Type>      activation,
    size_t                     thread_count,
    size_t               samples_per_worker
) :
    epochs(epochs), mini_batch_size(mini_batch_size), eta(eta), topology(topology),
    thread_state(thread_state), random_seed(random_seed), initializer(initializer), activation(activation),
    thread_count(thread_count), samples_per_worker(samples_per_worker)
    {
        param_options_count = {
            mini_batch_size.size(),
            eta.size(),
            topology.size(),
            thread_state.size(),
            random_seed.size(),
            initializer.size(),
            activation.size()
        };
    }

Observability::Observability(Config config)
{
    o_config = config;
    training_time_progression.reserve(o_config.epochs);
    testing_time_progression.reserve(o_config.epochs);
    accuracy_progression.reserve(o_config.epochs);
    epochs_to_accuracy_threshold = 0;
    time_to_accuracy_threshold = 0.0F;

    total_time = 0.0F;
    total_training_time = 0.0F;
    total_testing_time = 0.0F;
    avg_training_time = 0.0F;
    avg_testing_time = 0.0F;
    max_accuracy = 0.0F;
    final_accuracy = 0.0F;
}

void Observability::update(float training_time, float testing_time, float accuracy)
{
    training_time_progression.push_back(training_time);
    testing_time_progression.push_back(testing_time);
    accuracy_progression.push_back(accuracy);
}

void Observability::process()
{
    total_training_time =
        std::accumulate(training_time_progression.begin(),
                        training_time_progression.end(), 0.0F);
    total_testing_time =
        std::accumulate(testing_time_progression.begin(),
                        testing_time_progression.end(), 0.0F);
    total_time = total_training_time + total_testing_time;

    if (o_config.epochs == 0 || accuracy_progression.empty())
    {
        avg_training_time = avg_testing_time = max_accuracy = final_accuracy = 0.0F;
        epochs_to_accuracy_threshold = 0;
        time_to_accuracy_threshold = 0.0F;
        return;
    }

    const size_t completed_epochs = training_time_progression.size();
    avg_training_time = total_training_time / completed_epochs;
    avg_testing_time = total_testing_time / completed_epochs;
    max_accuracy = *std::max_element(accuracy_progression.begin(), accuracy_progression.end());
    final_accuracy = accuracy_progression.back();
    const int threshold_idx = find_idx_over_threshold(accuracy_progression, o_config.acc_threshold);
    epochs_to_accuracy_threshold = threshold_idx < 0 ? 0 : threshold_idx + 1;
    time_to_accuracy_threshold = std::accumulate(training_time_progression.begin(),
                                                  training_time_progression.begin() + epochs_to_accuracy_threshold,
                                                  0.0F) +
                                 std::accumulate(testing_time_progression.begin(),
                                                 testing_time_progression.begin() + epochs_to_accuracy_threshold,
                                                 0.0F);
}

void Observability::print_results()
{
    std::cout << "====== Benchmark Results ======\n"
              << "Configuration\n"
              << "  Epochs: " << o_config.epochs << '\n'
              << "  Mini-batch size: " << o_config.mini_batch_size << '\n'
              << "  Learning rate (eta): " << o_config.eta << '\n'
              << "  Topology: ";
    for (size_t i = 0; i < o_config.topology.size(); ++i)
        std::cout << (i == 0 ? "" : " -> ") << o_config.topology[i];

    std::cout << "\n  Threading: "
              << (o_config.thread_state == Threading::Multi ? "Multi" : "Single")
              << "\n  Thread count: " << o_config.thread_count << '\n'
              << "Results (seconds unless noted)\n"
              << std::fixed << std::setprecision(3)
              << "  Total time: " << total_time << '\n'
              << "  Total training time: " << total_training_time << '\n'
              << "  Total testing time: " << total_testing_time << '\n'
              << "  Average training time/epoch: " << avg_training_time << '\n'
              << "  Average testing time/epoch: " << avg_testing_time << '\n'
              << std::setprecision(2)
              << "  Maximum accuracy: " << max_accuracy << "%\n"
              << "  Final accuracy: " << final_accuracy << "%\n"
              << "  Accuracy threshold: " << o_config.acc_threshold << "%\n";

    if (epochs_to_accuracy_threshold == 0)
        std::cout << "  Epochs to threshold: not reached\n";
    else
        std::cout << "  Epochs to threshold: " << epochs_to_accuracy_threshold << '\n'
                  << std::setprecision(3)
                  << "  Time to threshold: " << time_to_accuracy_threshold << " s\n";
    std::cout << std::defaultfloat;
}

json Observability::to_json_object()
{
    return json{
        {"config", o_config.to_json_object()},
        {"run results",
            {
                {"training_time_progression", training_time_progression},
                {"testing_time_progression", testing_time_progression},
                {"accuracy_progression", accuracy_progression},
                {"epochs_to_accuracy_threshold", epochs_to_accuracy_threshold},
                {"time_to_accuracy_threshold", time_to_accuracy_threshold},
                {"total_time", total_time},
                {"total_training_time", total_training_time},
                {"total_testing_time", total_testing_time},
                {"avg_training_time", avg_training_time},
                {"avg_testing_time", avg_testing_time},
                {"max_accuracy", max_accuracy},
                {"final_accuracy", final_accuracy}
            }
        }
    };
}

Benchmark::Benchmark(BenchConfig b_config_)
    : b_config(b_config_), odo(b_config.param_options_count) {}

void Benchmark::run() {
    if (b_config.epochs == 0 || b_config.epochs > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        b_config.thread_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        b_config.samples_per_worker == 0 || b_config.samples_per_worker > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid benchmark epochs or worker settings.");
    for (auto count : b_config.param_options_count)
        if (count == 0) throw std::invalid_argument("Benchmark parameter lists must not be empty.");
    for (auto batch : b_config.mini_batch_size)
        if (batch <= 0) throw std::invalid_argument("Mini-batch sizes must be positive.");
    for (auto eta : b_config.eta)
        if (!(eta > 0) || (std::bit_cast<uint32_t>(eta) & 0x7f800000U) == 0x7f800000U) throw std::invalid_argument("Learning rates must be finite and positive.");
    for (auto activation : b_config.activation)
        if (static_cast<unsigned>(activation) > static_cast<unsigned>(act::Type::LeakyRelu))
            throw std::invalid_argument("Invalid activation.");
    for (auto initializer : b_config.initializer)
        if (static_cast<unsigned>(initializer) > static_cast<unsigned>(init::Type::LeCunNormal))
            throw std::invalid_argument("Invalid initializer.");
    for (auto threading : b_config.thread_state)
        if (threading != Threading::Single && threading != Threading::Multi)
            throw std::invalid_argument("Invalid threading mode.");
    for (const auto& topology : b_config.topology)
        if (topology.size() < 2 || topology.front() != 784 || topology.back() != 10 ||
            std::any_of(topology.begin(), topology.end(), [](int n) { return n <= 0; }))
            throw std::invalid_argument("Benchmark requires a positive MNIST topology (784 -> ... -> 10).");

    const auto now = std::chrono::system_clock::now();
    const auto timestamp = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::ostringstream filename;
    filename << "results_" << std::put_time(std::gmtime(&timestamp), "%Y-%m-%d_%H-%M-%S")
             << '_' << std::setfill('0') << std::setw(3) << milliseconds << "_UTC.jsonl";
    if (std::filesystem::exists(filename.str()))
        throw std::runtime_error("Result filename already exists: " + filename.str());

    Timer timer;

    auto training_data = std::make_shared<const TrainingData>(
        MNIST_loader::load_training_data(config.train_images, config.train_labels, 50000, false));
    auto test_data = std::make_shared<const TestData>(
        MNIST_loader::load_test_data(config.test_images, config.test_labels, 10000));

    std::ofstream file(filename.str());
    if (!file) throw std::runtime_error("Could not open " + filename.str() + " for writing.");

    std::cout << "Results file: " << filename.str() << '\n';
    file.exceptions(std::ios::badbit | std::ios::failbit);
    int i = 0;

    do {
        i++;
        auto state = odo.get_state();
        config = {
            static_cast<int>(b_config.epochs),
            b_config.mini_batch_size[state[0]],
            b_config.eta[state[1]],
            b_config.topology[state[2]],
            b_config.thread_state[state[3]],
            b_config.random_seed[state[4]],
            b_config.initializer[state[5]],
            b_config.activation[state[6]],
        };
        config.thread_count = static_cast<int>(b_config.thread_count);
        config.samples_per_worker = static_cast<int>(b_config.samples_per_worker);
        std::cout << "Config no = " << i << ", Time = " << timer.elapsed() << "\n";
        print_container(state);

        Network net(config, training_data, test_data);
        std::string error;
        try { net.SGD(false); }
        catch (const std::runtime_error& e) { error = e.what(); net.observe.process(); }

        auto record = net.observe.to_json_object();
        record["evaluation_method"] = "batched_128_v1";
        record["status"] = error.empty() ? "completed" : "failed";
        if (!error.empty()) record["error"] = error;
        const int hardware_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        record["effective_thread_count"] = config.thread_state == Threading::Single ? 1 :
            std::min(config.thread_count == 0 ? hardware_threads : std::min(hardware_threads, config.thread_count),
                1 + (config.mini_batch_size - 1) / config.samples_per_worker);
        record["training_samples"] = training_data->size();
        record["test_samples"] = test_data->size();
        record["config_no"] = i;
        file << record.dump() << '\n' << std::flush;

    } while (odo.next());
}
