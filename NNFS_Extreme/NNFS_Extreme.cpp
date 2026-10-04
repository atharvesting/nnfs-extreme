// #include "examples.hpp"
#include <iostream>
#include "benchmark_harness.hpp"
#include "examples.hpp"

BenchConfig bc(
    30,
    {32},
    {1.0F},
    {{784, 30, 10}},
    {Threading::Single, Threading::Multi},
    {21},
    {init::Type::XavierNormal},
    {act::Type::Sigmoid},
    16
);

int main()
{
    BenchConfig b_config; // Default Initialization to full suite
    Benchmark bench(b_config);
    std::cout << "Benchmark Harness initialized!\n";

    bench.run({2, 1, 2, 0, 2, 0, 0});
    std::cout << "Benchmark run complete!" << std::endl;
    // examples::standard_mnist();

	return 0;
}
