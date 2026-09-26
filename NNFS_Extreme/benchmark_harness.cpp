#include "benchmark_harness.hpp"
#include "utils.hpp"
#include <algorithm> // max
#include <numeric>   // accumulate
#include <iostream>
#include <iomanip>

Config::Config(int epochs, int mini_batch_size, float eta,
               std::vector<int> topology, Threading thread_state,
               int thread_count, std::string train_images,
               std::string test_images, std::string train_labels,
               std::string test_labels, float acc_threshold)

    : epochs(epochs), mini_batch_size(mini_batch_size), eta(eta),
      topology(topology), thread_state(thread_state),
      thread_count(thread_count), train_images(train_images),
      test_images(test_images), train_labels(train_labels),
      test_labels(test_labels), acc_threshold(acc_threshold) {}

BenchConfig::BenchConfig(
    size_t                           epochs,
    std::vector<size_t>     mini_batch_size,
    std::vector<float>                  eta,
    std::vector<std::vector<int>>  topology,
    std::vector<Threading>     thread_state,
    std::vector<int>            random_seed,
    std::vector<bool>           xavier_init,
    size_t                     thread_count
) :
    epochs(epochs), mini_batch_size(mini_batch_size), eta(eta), topology(topology),
    thread_state(thread_state), random_seed(random_seed), xavier_init(xavier_init),
    thread_count(thread_count) {}

Observability::Observability(Config config)
{
    o_config = config;
    epochs = config.epochs;
    training_time_progression.reserve(epochs);
    testing_time_progression.reserve(epochs);
    accuracy_progression.reserve(epochs);
    acc_threshold = config.acc_threshold;
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

    if (epochs == 0 || accuracy_progression.empty())
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
    const int threshold_idx = find_idx_over_threshold(accuracy_progression, acc_threshold);
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
              << "  Accuracy threshold: " << acc_threshold << "%\n";

    if (epochs_to_accuracy_threshold == 0)
        std::cout << "  Epochs to threshold: not reached\n";
    else
        std::cout << "  Epochs to threshold: " << epochs_to_accuracy_threshold << '\n'
                  << std::setprecision(3)
                  << "  Time to threshold: " << time_to_accuracy_threshold << " s\n";
    std::cout << std::defaultfloat;
}
