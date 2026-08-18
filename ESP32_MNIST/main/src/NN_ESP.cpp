#include "NN_ESP.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

template <typename T>
Matrix<T>::Matrix() : rows(0), cols(0), rix() {}

template <typename T>
Matrix<T>::Matrix(size_t rows_, size_t cols_, const T& value)
    : rows(rows_), cols(cols_), rix(rows_ * cols_, value) {}

template <typename T>
Matrix<T>::Matrix(size_t rows_, size_t cols_, const std::vector<T>& values)
    : rows(rows_), cols(cols_), rix(values) {
    if (rix.size() != rows_ * cols_) {
        throw std::invalid_argument("Matrix dimensions and rix size do not match");
    }
}

template <typename T>
size_t Matrix<T>::size() const {
    return rows * cols;
}

template <typename T>
void Matrix<T>::fill(T value) {
    for (auto& element : rix) {
        element = value;
    }
}

template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return rix[row * cols + col];
}

template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
    return rix[row * cols + col];
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix addition requires same dimensions");
    }

    Matrix<T> result(rows, cols, T{});
    for (size_t i = 0; i < rix.size(); ++i) {
        result.rix[i] = rix[i] + other.rix[i];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix subtraction requires same dimensions");
    }

    Matrix<T> result(rows, cols, T{});
    for (size_t i = 0; i < rix.size(); ++i) {
        result.rix[i] = rix[i] - other.rix[i];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::hadamard(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Hadamard product requires same dimensions");
    }

    Matrix<T> result(rows, cols, T{});
    for (size_t i = 0; i < rix.size(); ++i) {
        result.rix[i] = rix[i] * other.rix[i];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::matmul(const Matrix<T>& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    Matrix<T> result(rows, other.cols, T{});
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < other.cols; ++c) {
            T sum = T{};
            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(r, k) * other(k, c);
            }
            result(r, c) = sum;
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols, rows, T{});
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

template <typename T, typename U, typename V>
void gemm(float alpha, const Matrix<T>& A, const Matrix<U>& B, float beta, Matrix<V>& C) {

	if (beta != 1.0F) for (auto& n : C.rix) n *= beta;

	typename std::vector<T>::const_iterator ia = A.rix.begin();
	typename std::vector<U>::const_iterator ib = B.rix.begin();
	typename std::vector<V>::iterator ic = C.rix.begin();

	constexpr size_t block_size = 16;

	for (size_t bi = 0; bi < A.rows; bi += block_size) {
		for (size_t bj = 0; bj < B.cols; bj += block_size) {
			for (size_t bk = 0; bk < A.cols; bk += block_size) {

				size_t i_max = std::min(bi + block_size, A.rows);
				size_t j_max = std::min(bj + block_size, B.cols);
				size_t k_max = std::min(bk + block_size, A.cols);
				
				for (size_t i = bi; i < i_max; i++) {
					for (size_t k = bk; k < k_max; k++) {
						auto a_scaled = static_cast<V>(alpha * ia[i * A.cols + k]);

						for (size_t j = bj; j < j_max; j++) {
							ic[i * C.cols + j] += a_scaled * ib[k * B.cols + j];
						}
					}
				}
			}
		}
	}
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

Matrix<float> sigmoid(const Matrix<float>& input) {
    Matrix<float> output(input.rows, input.cols, 0.0f);
    for (size_t i = 0; i < input.rix.size(); ++i) {
        output.rix[i] = sigmoid(input.rix[i]);
    }
    return output;
}

// Explicit template instantiations for the ESP32 build.
template class Matrix<float>;
template class Matrix<int>;
template class Matrix<uint8_t>;

template <typename T>
Matrix<T> feedforward(const Matrix<T>& m,
                      const std::vector<Matrix<T>>& weights,
                      const std::vector<Matrix<T>>& biases,
                      size_t num_param_layers)
{
    if (weights.size() != biases.size() || weights.size() < num_param_layers) {
        throw std::invalid_argument("weights and biases must match and cover the requested layers");
    }

    Matrix<T> output = m;
    for (size_t i = 0; i < num_param_layers; ++i) {
        if (output.rows != weights[i].cols) {
            throw std::invalid_argument("Input height does not match weight columns");
        }

        Matrix<T> linear = weights[i].matmul(output) + biases[i];
        output = sigmoid(linear);
    }

    return output;
}

template <typename T>
int _argmax(const Matrix<T>& output) {
    if (output.rix.empty()) {
        throw std::invalid_argument("Cannot argmax an empty matrix");
    }

    auto max_it = std::max_element(output.rix.begin(), output.rix.end());
    size_t argmax = std::distance(output.rix.begin(), max_it);
    return static_cast<int>(argmax);
}

template Matrix<float> feedforward<float>(const Matrix<float>&, const std::vector<Matrix<float>>&, const std::vector<Matrix<float>>&, size_t);
template int _argmax<float>(const Matrix<float>&);