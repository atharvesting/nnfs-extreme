// #include "examples.hpp"
#include <iostream>
#include "benchmark_harness.hpp"

BenchConfig bc(
    30,
    {32},
    {1.0F},
    {{784, 30, 10}},
    {Threading::Single, Threading::Multi},
    {21},
    {true},
    16
);

int main()
{
    BenchConfig b_config;
    Benchmark bench(b_config);
    std::cout << "Benchmark Harness initialized!\n";

    bench.run();
    std::cout << "Benchmark run complete!" << std::endl;

	return 0;
}
