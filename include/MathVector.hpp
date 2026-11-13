#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace miniann {

struct MathVector {
    std::vector<double> data;

    MathVector() = default;
    MathVector(const std::vector<double>& d) : data(d) {}
    explicit MathVector(int size, double initial_value = 0.0) {
        if (size < 0) {
            throw std::invalid_argument("MathVector size cannot be negative.");
        }
        data.assign(static_cast<std::size_t>(size), initial_value);
    }

    std::size_t size() const noexcept { return data.size(); }

    // Dot Product
    double operator*(const MathVector& other) const {
        if (this->data.size() != other.data.size()) {
            throw std::invalid_argument("Size mismatch for dot product.");
        }
        double result = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            result += this->data[i] * other.data[i];
        }
        return result;
    }

    // Scalar Multiplication
    MathVector operator*(double scalar) const {
        std::vector<double> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return MathVector(result);
    }

    // Element-wise Subtraction
    MathVector operator-(const MathVector& other) const {
        if (this->data.size() != other.data.size()) {
            throw std::invalid_argument("Size mismatch for vector subtraction.");
        }
        std::vector<double> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = this->data[i] - other.data[i];
        }
        return MathVector(result);
    }

    // Element-wise Addition
    MathVector operator+(const MathVector& other) const {
        if (this->data.size() != other.data.size()) {
            throw std::invalid_argument("Size mismatch for vector addition.");
        }
        std::vector<double> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = this->data[i] + other.data[i];
        }
        return MathVector(result);
    }
};

} // namespace miniann
