#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include "MathVector.hpp"
#include "Matrix.hpp"

namespace miniann {

class IActivation {
public:
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::string name() const = 0;
    virtual double weightInitStdDev(int fan_in, int fan_out) const = 0;

    virtual bool isLayerLevel() const { return false; }

    virtual void apply(const Matrix& Z, Matrix& A) const {
        A.reallocate(Z.rows(), Z.cols());
        const std::size_t total = Z.size();
        const double* zPtr = Z.dataPtr();
        double* aPtr = A.dataPtr();
        for (std::size_t i = 0; i < total; ++i) {
            aPtr[i] = activate(zPtr[i]);
        }
    }

    virtual void applyDerivative(const Matrix& Z, const Matrix& /*A*/, const Matrix& upstreamGrad, Matrix& delta) const {
        delta.reallocate(Z.rows(), Z.cols());
        const std::size_t total = Z.size();
        const double* zPtr = Z.dataPtr();
        const double* upPtr = upstreamGrad.dataPtr();
        double* deltaPtr = delta.dataPtr();
        for (std::size_t i = 0; i < total; ++i) {
            deltaPtr[i] = upPtr[i] * derivative(zPtr[i]);
        }
    }

    virtual ~IActivation() = default;
};

class Sigmoid : public IActivation {
public:
    double activate(double x) const override {
        if (x >= 0.0) {
            double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        double z = std::exp(x);
        return z / (1.0 + z);
    }
    double derivative(double x) const override {
        double sig = activate(x);
        return sig * (1.0 - sig);
    }
    std::string name() const override { return "Sigmoid"; }
    double weightInitStdDev(int fan_in, int fan_out) const override {
        return std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    }
};

class ReLU : public IActivation {
public:
    double activate(double x) const override { return (x > 0.0) ? x : 0.0; }
    double derivative(double x) const override { return (x > 0.0) ? 1.0 : 0.0; }
    std::string name() const override { return "ReLU"; }
    double weightInitStdDev(int fan_in, int /*fan_out*/) const override {
        return std::sqrt(2.0 / static_cast<double>(fan_in));
    }
};

class Tanh : public IActivation {
public:
    double activate(double x) const override { return std::tanh(x); }
    double derivative(double x) const override {
        double t = std::tanh(x);
        return 1.0 - (t * t);
    }
    std::string name() const override { return "Tanh"; }
    double weightInitStdDev(int fan_in, int fan_out) const override {
        return std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    }
};

class Softmax : public IActivation {
public:
    double activate(double /*x*/) const override {
        throw std::runtime_error("Softmax::activate(double) is unsupported; Softmax requires layer-level vector/matrix.");
    }
    double derivative(double /*x*/) const override {
        throw std::runtime_error("Softmax::derivative(double) is unsupported; Softmax requires layer-level vector/matrix.");
    }
    std::string name() const override { return "Softmax"; }
    bool isLayerLevel() const override { return true; }

    double weightInitStdDev(int fan_in, int fan_out) const override {
        return std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    }

    void apply(const Matrix& Z, Matrix& A) const override {
        A.reallocate(Z.rows(), Z.cols());
        const std::size_t rows = Z.rows();
        const std::size_t cols = Z.cols();

        for (std::size_t r = 0; r < rows; ++r) {
            const double* zRow = Z.rowPtr(r);
            double* aRow = A.rowPtr(r);

            double maxVal = zRow[0];
            for (std::size_t c = 1; c < cols; ++c) {
                if (zRow[c] > maxVal) maxVal = zRow[c];
            }

            double sumExp = 0.0;
            for (std::size_t c = 0; c < cols; ++c) {
                aRow[c] = std::exp(zRow[c] - maxVal);
                sumExp += aRow[c];
            }

            const double invSum = 1.0 / (sumExp > 0.0 ? sumExp : 1e-15);
            for (std::size_t c = 0; c < cols; ++c) {
                aRow[c] *= invSum;
            }
        }
    }

    void applyDerivative(const Matrix& /*Z*/, const Matrix& A, const Matrix& upstreamGrad, Matrix& delta) const override {
        delta.reallocate(A.rows(), A.cols());
        const std::size_t rows = A.rows();
        const std::size_t cols = A.cols();

        for (std::size_t r = 0; r < rows; ++r) {
            const double* aRow = A.rowPtr(r);
            const double* upRow = upstreamGrad.rowPtr(r);
            double* dRow = delta.rowPtr(r);

            double dot = 0.0;
            for (std::size_t c = 0; c < cols; ++c) {
                dot += upRow[c] * aRow[c];
            }

            for (std::size_t c = 0; c < cols; ++c) {
                dRow[c] = aRow[c] * (upRow[c] - dot);
            }
        }
    }
};

inline std::shared_ptr<const IActivation> createActivation(const std::string& activationName) {
    if (activationName == "Sigmoid") return std::make_shared<Sigmoid>();
    if (activationName == "ReLU") return std::make_shared<ReLU>();
    if (activationName == "Tanh") return std::make_shared<Tanh>();
    if (activationName == "Softmax") return std::make_shared<Softmax>();
    throw std::invalid_argument("Unknown activation function: " + activationName);
}

} // namespace miniann