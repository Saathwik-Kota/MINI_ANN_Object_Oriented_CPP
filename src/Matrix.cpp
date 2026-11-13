#include "Matrix.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miniann {

void Matrix::addBiasRow(const std::vector<double>& bias) {
    if (bias.size() != cols_) {
        throw std::invalid_argument(
            "Bias size (" + std::to_string(bias.size()) +
            ") does not match Matrix column count (" + std::to_string(cols_) + ").");
    }

    const std::size_t rCount = rows_;
    const std::size_t cCount = cols_;
    const double* bPtr = bias.data();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((rCount * cCount) >= 8192)
#endif
    for (long long i = 0; i < static_cast<long long>(rCount); ++i) {
        double* row = data_.data() + (static_cast<std::size_t>(i) * cCount);
        for (std::size_t j = 0; j < cCount; ++j) {
            row[j] += bPtr[j];
        }
    }
}

void Matrix::sumColumns(std::vector<double>& out) const {
    out.assign(cols_, 0.0);
    const std::size_t rCount = rows_;
    const std::size_t cCount = cols_;
    double* outPtr = out.data();

    for (std::size_t i = 0; i < rCount; ++i) {
        const double* row = data_.data() + (i * cCount);
        for (std::size_t j = 0; j < cCount; ++j) {
            outPtr[j] += row[j];
        }
    }
}

void Matrix::gemm(const Matrix& A, const Matrix& B, Matrix& C,
                  bool transA, bool transB,
                  double alpha, double beta) {
    const std::size_t M = transA ? A.cols_ : A.rows_;
    const std::size_t K_A = transA ? A.rows_ : A.cols_;
    const std::size_t K_B = transB ? B.cols_ : B.rows_;
    const std::size_t N = transB ? B.rows_ : B.cols_;

    if (K_A != K_B) {
        throw std::invalid_argument(
            "GEMM inner dimension mismatch: A has " + std::to_string(K_A) +
            " while B has " + std::to_string(K_B) + ".");
    }

    C.reallocate(M, N);
    const std::size_t K = K_A;
    const std::size_t totalOps = M * N * K;

    if (beta == 0.0) {
        std::fill(C.data_.begin(), C.data_.end(), 0.0);
    } else if (beta != 1.0) {
        for (std::size_t idx = 0; idx < C.data_.size(); ++idx) {
            C.data_[idx] *= beta;
        }
    }

    if (alpha == 0.0 || M == 0 || N == 0 || K == 0) {
        return;
    }

    const double* aData = A.data_.data();
    const double* bData = B.data_.data();
    double* cData = C.data_.data();
    const std::size_t aCols = A.cols_;
    const std::size_t bCols = B.cols_;
    const std::size_t cCols = C.cols_;

    if (!transA && !transB) {
        // C(M, N) += alpha * A(M, K) * B(K, N) -> i-k-j unit-stride access
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (totalOps >= 8192 && M >= 2)
#endif
        for (long long i = 0; i < static_cast<long long>(M); ++i) {
            double* cRow = cData + (static_cast<std::size_t>(i) * cCols);
            const double* aRow = aData + (static_cast<std::size_t>(i) * aCols);
            for (std::size_t k = 0; k < K; ++k) {
                const double a_ik = alpha * aRow[k];
                const double* bRow = bData + (k * bCols);
                for (std::size_t j = 0; j < N; ++j) {
                    cRow[j] += a_ik * bRow[j];
                }
            }
        }
    } else if (!transA && transB) {
        // C(M, N) += alpha * A(M, K) * B^T(K, N), B is (N, K) -> both rows contiguous
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (totalOps >= 8192)
#endif
        for (long long i = 0; i < static_cast<long long>(M); ++i) {
            const double* aRow = aData + (static_cast<std::size_t>(i) * aCols);
            double* cRow = cData + (static_cast<std::size_t>(i) * cCols);
            for (std::size_t j = 0; j < N; ++j) {
                const double* bRow = bData + (j * bCols);
                double dot = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    dot += aRow[k] * bRow[k];
                }
                cRow[j] += alpha * dot;
            }
        }
    } else if (transA && !transB) {
        // C(M, N) += alpha * A^T(M, K) * B(K, N), A is (K, M) -> parallelize over output rows i
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (totalOps >= 8192 && M >= 2)
#endif
        for (long long i = 0; i < static_cast<long long>(M); ++i) {
            double* cRow = cData + (static_cast<std::size_t>(i) * cCols);
            for (std::size_t k = 0; k < K; ++k) {
                const double a_ki = alpha * aData[k * aCols + i];
                const double* bRow = bData + (k * bCols);
                for (std::size_t j = 0; j < N; ++j) {
                    cRow[j] += a_ki * bRow[j];
                }
            }
        }
    } else {
        // C(M, N) += alpha * A^T(M, K) * B^T(K, N), A is (K, M), B is (N, K)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (totalOps >= 8192)
#endif
        for (long long i = 0; i < static_cast<long long>(M); ++i) {
            double* cRow = cData + (static_cast<std::size_t>(i) * cCols);
            for (std::size_t j = 0; j < N; ++j) {
                const double* bRow = bData + (j * bCols);
                double dot = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    dot += aData[k * aCols + i] * bRow[k];
                }
                cRow[j] += alpha * dot;
            }
        }
    }
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    Matrix res(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        res.data_[i] = data_[i] + other.data_[i];
    }
    return res;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }
    Matrix res(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        res.data_[i] = data_[i] - other.data_[i];
    }
    return res;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix res(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        res.data_[i] = data_[i] * scalar;
    }
    return res;
}

} // namespace miniann

