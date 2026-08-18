#pragma once

#include <Arduino.h>
#include <FS.h>
#include <stdint.h>
#include <vector>
#include <string>

// Minimal serial-only subset of the Spalten matrix API needed for ESP32 inference.
template <typename T>
class Matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<T> rix;

    Matrix();
    Matrix(size_t rows_, size_t cols_, const T& value = T{});
    Matrix(size_t rows_, size_t cols_, const std::vector<T>& values);

    size_t size() const;
    void fill(T value);
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;

    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> hadamard(const Matrix<T>& other) const;
    Matrix<T> matmul(const Matrix<T>& other) const;
    Matrix<T> transpose() const;
};

float sigmoid(float value);
Matrix<float> sigmoid(const Matrix<float>& input);

template <typename T>
Matrix<T> feedforward(const Matrix<T>& m,
                      const std::vector<Matrix<T>>& weights,
                      const std::vector<Matrix<T>>& biases,
                      size_t num_param_layers);


template <typename T>
int _argmax(const Matrix<T>& output);