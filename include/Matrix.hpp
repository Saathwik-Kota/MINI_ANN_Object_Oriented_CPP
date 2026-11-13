#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace miniann {

class Matrix {
private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<double> data_;

public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double initVal = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, initVal) {}

    Matrix(std::size_t rows, std::size_t cols, const std::vector<double>& data)
        : rows_(rows), cols_(cols), data_(data) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument(
                "Matrix data size (" + std::to_string(data.size()) +
                ") does not match dimensions (" + std::to_string(rows) +
                "x" + std::to_string(cols) + ").");
        }
    }

    Matrix(std::size_t rows, std::size_t cols, std::vector<double>&& data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows * cols) {
            throw std::invalid_argument("Matrix dimension mismatch with rvalue buffer.");
        }
    }

    static Matrix from2D(const std::vector<std::vector<double>>& grid) {
        if (grid.empty()) return Matrix();
        const std::size_t r = grid.size();
        const std::size_t c = grid[0].size();
        Matrix m(r, c);
        for (std::size_t i = 0; i < r; ++i) {
            if (grid[i].size() != c) {
                throw std::invalid_argument("Ragged 2D grid cannot be converted to Matrix.");
            }
            std::copy(grid[i].begin(), grid[i].end(), m.rowPtr(i));
        }
        return m;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    double& operator()(std::size_t r, std::size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    const double& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    double* dataPtr() noexcept { return data_.data(); }
    const double* dataPtr() const noexcept { return data_.data(); }

    std::vector<double>& data() noexcept { return data_; }
    const std::vector<double>& data() const noexcept { return data_; }

    double* rowPtr(std::size_t r) noexcept { return data_.data() + (r * cols_); }
    const double* rowPtr(std::size_t r) const noexcept { return data_.data() + (r * cols_); }

    void reshape(std::size_t newRows, std::size_t newCols) {
        if (newRows * newCols != data_.size()) {
            throw std::invalid_argument("Cannot reshape Matrix: total element count must remain equal.");
        }
        rows_ = newRows;
        cols_ = newCols;
    }

    void resize(std::size_t newRows, std::size_t newCols, double initVal = 0.0) {
        rows_ = newRows;
        cols_ = newCols;
        data_.assign(newRows * newCols, initVal);
    }

    void reallocate(std::size_t newRows, std::size_t newCols) {
        rows_ = newRows;
        cols_ = newCols;
        if (data_.size() != newRows * newCols) {
            data_.resize(newRows * newCols);
        }
    }

    void fill(double val) noexcept {
        std::fill(data_.begin(), data_.end(), val);
    }

    void extractRow(std::size_t r, std::vector<double>& out) const {
        if (r >= rows_) throw std::out_of_range("Row index out of range.");
        out.resize(cols_);
        const double* ptr = rowPtr(r);
        std::copy(ptr, ptr + cols_, out.begin());
    }

    void setRow(std::size_t r, const double* src) {
        if (r >= rows_) throw std::out_of_range("Row index out of range.");
        double* dest = rowPtr(r);
        for (std::size_t j = 0; j < cols_; ++j) {
            dest[j] = src[j];
        }
    }

    void addBiasRow(const std::vector<double>& bias);
    void sumColumns(std::vector<double>& out) const;

    static void gemm(const Matrix& A, const Matrix& B, Matrix& C,
                     bool transA = false, bool transB = false,
                     double alpha = 1.0, double beta = 0.0);

    static Matrix multiply(const Matrix& A, const Matrix& B,
                           bool transA = false, bool transB = false) {
        const std::size_t m = transA ? A.cols() : A.rows();
        const std::size_t n = transB ? B.rows() : B.cols();
        Matrix C(m, n);
        gemm(A, B, C, transA, transB, 1.0, 0.0);
        return C;
    }

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
};

} // namespace miniann
