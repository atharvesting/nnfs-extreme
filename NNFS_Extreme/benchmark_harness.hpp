#pragma once
#include <cstddef>
#include <vector>   // vector
#include <string>   // string
#include <nlohmann/json.hpp>
#include "utils.hpp"

using json = nlohmann::json;

enum class Threading { Single, Multi };

struct Config {
    int epochs;
    int mini_batch_size;
    float eta;
    std::vector<int> topology;
    Threading thread_state;
    int thread_count;
    std::string train_images;
    std::string test_images;
    std::string train_labels;
    std::string test_labels;
    float acc_threshold;
    int random_seed;
    bool xavier_init;

    Config(int epochs = 30, int mini_batch_size = 128, float eta = 30.0F,
        std::vector<int>  topology = {784, 30, 10},
        Threading     thread_state = Threading::Single,
        int            random_seed = 42,
        bool           xavier_init = true,
        int           thread_count = 16,
        std::string   train_images = "data/mnist_train_images.bin",
        std::string    test_images = "data/mnist_test_images.bin",
        std::string   train_labels = "data/mnist_train_labels.bin",
        std::string    test_labels = "data/mnist_test_labels.bin",
        float        acc_threshold = 94.0F
    );

    json to_json_object();
};

struct Observability
{
    // Independent
                Config o_config;
    std::vector<float> training_time_progression;
    std::vector<float> testing_time_progression;
    std::vector<float> accuracy_progression;
                size_t epochs_to_accuracy_threshold;
                 float time_to_accuracy_threshold;
    // Dependent
                 float total_time;
                 float total_training_time;
                 float total_testing_time;
                 float avg_training_time;
                 float avg_testing_time;
                 float max_accuracy;
                 float final_accuracy;

    Observability(Config config);
    void update(float training_time, float testing_time, float accuracy);
    void process();
    void print_results();
    json to_json_object();
};


struct BenchConfig
{
    size_t epochs;                              // Fixed
    std::vector<int> mini_batch_size;           // 4
    std::vector<float> eta;                     // 4
    std::vector<std::vector<int>> topology;     // 3
    std::vector<Threading> thread_state;        // 2
    std::vector<int> random_seed;               // 4
    std::vector<bool> xavier_init;              // 2
    size_t thread_count;                        // Fixed
    std::vector<size_t> param_options_count;    // Derived

    BenchConfig(
        size_t                           epochs = 30,
        std::vector<int>        mini_batch_size = {32, 64, 128, 256},
        std::vector<float>                  eta = {1.0F, 3.0F, 6.0F, 9.0F},
        std::vector<std::vector<int>>  topology = {{784, 30, 10}, {784, 128, 30, 10}, {784, 512, 512, 10}},
        std::vector<Threading>     thread_state = {Threading::Single, Threading::Multi},
        std::vector<int>            random_seed = {21, 42, 84, 168},
        std::vector<bool>           xavier_init = {true, false},
        size_t                     thread_count = 16
    );
};

// template <typename T>
// struct MemberMap
// {
//     std::vector<T> BenchConfig::* source;
//     T Config::* destination;
// };

class Benchmark {
private:
    BenchConfig b_config;
    Odometer odo;
    Config config;
    json result;

    // inline static constexpr auto maps = std::make_tuple(
    //     MemberMap<float>{ &BenchConfig::eta, &Config::eta },
    //     MemberMap<int>{ &BenchConfig::mini_batch_size, &Config::mini_batch_size },
    //     MemberMap<std::vector<int>>{ &BenchConfig::topology, &Config::topology },
    //     MemberMap<Threading>{ &BenchConfig::thread_state, &Config::thread_state },
    //     MemberMap<int>{ &BenchConfig::random_seed, &Config::random_seed },
    //     MemberMap<bool>{ &BenchConfig::xavier_init, &Config::xavier_init }
    // );

public:
    Benchmark(BenchConfig b_config_);
    void run();
};
