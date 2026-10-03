#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
check_dir=$(mktemp -d)
trap 'rm -rf "$check_dir"' EXIT
g++ -std=c++23 -O3 -DNDEBUG -march=native -ffast-math -pthread \
    -I"$root/NNFS_Extreme" -I"$root/third_party/Spalten/include" \
    -I"$root/build/linux-gcc-release/_deps/nlohmann_json-src/include" \
    "$root/scripts/check_benchmark.cpp" \
    "$root/NNFS_Extreme/NN.cpp" "$root/NNFS_Extreme/benchmark_harness.cpp" \
    "$root/NNFS_Extreme/data_loaders.cpp" "$root/NNFS_Extreme/activation_functions.cpp" \
    "$root/NNFS_Extreme/utils.cpp" -ltbb -o "$check_dir/check"
ln -s "$root/data" "$check_dir/data"
cd "$check_dir"
./check
