#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include "MathVector.hpp"
#include "Matrix.hpp"

namespace miniann {

class ILossFunction {
public:
    virtual double calculate(double prediction, double target) const = 0;
    virtual double derivative(double prediction, double target) const = 0;
    virtual std::string name() const = 0;
    virtual bool isMatrixLoss() const { return false; }

    virtual double calculateBatch(const Matrix& predictions, const Matrix& targets) const {
        const std::size_t count = predictions.size();
        const double* p = predictions.dataPtr();
        const double* t = targets.dataPtr();
        double total = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            total += calculate(p[i], t[i]);
        }
        return total;
    }

    virtual void derivativeBatch(const Matrix& predictions, const Matrix& targets, Matrix& dL_da) const {
        dL_da.reallocate(predictions.rows(), predictions.cols());
        const std::size_t count = predictions.size();
        const double* p = predictions.dataPtr();
        const double* t = targets.dataPtr();
        double* dPtr = dL_da.dataPtr();
        for (std::size_t i = 0; i < count; ++i) {
            dPtr[i] = derivative(p[i], t[i]);
        }
    }

    virtual ~ILossFunction() = default;
};

class MSE : public ILossFunction {
public:
    double calculate(double prediction, double target) const override {
        return 0.5 * std::pow((prediction - target), 2);
    }
    double derivative(double prediction, double target) const override {
        return (prediction - target);
    }
    std::string name() const override { return "MSE"; }
};

class BinaryCrossEntropy : public ILossFunction {
public:
    double calculate(double prediction, double target) const override {
        constexpr double epsilon = 1e-15;
        prediction = std::max(epsilon, std::min(1.0 - epsilon, prediction));
        return -(target * std::log(prediction) + (1.0 - target) * std::log(1.0 - prediction));
    }

    double derivative(double prediction, double target) const override {
        constexpr double epsilon = 1e-15;
        prediction = std::max(epsilon, std::min(1.0 - epsilon, prediction));
        return -(target / prediction) + ((1.0 - target) / (1.0 - prediction));
    }
    std::string name() const override { return "BinaryCrossEntropy"; }
};

class CategoricalCrossEntropy : public ILossFunction {
public:
    bool isMatrixLoss() const override { return true; }
    std::string name() const override { return "CategoricalCrossEntropy"; }

    double calculate(double prediction, double target) const override {
        constexpr double epsilon = 1e-15;
        prediction = std::max(epsilon, std::min(1.0 - epsilon, prediction));
        return -(target * std::log(prediction));
    }

    double derivative(double prediction, double target) const override {
        constexpr double epsilon = 1e-15;
        prediction = std::max(epsilon, std::min(1.0 - epsilon, prediction));
        return -(target / prediction);
    }

    double calculateBatch(const Matrix& predictions, const Matrix& targets) const override {
        constexpr double epsilon = 1e-15;
        const std::size_t rows = predictions.rows();
        const std::size_t cols = predictions.cols();
        double totalLoss = 0.0;

        for (std::size_t r = 0; r < rows; ++r) {
            const double* pRow = predictions.rowPtr(r);
            const double* tRow = targets.rowPtr(r);
            for (std::size_t c = 0; c < cols; ++c) {
                if (tRow[c] > 0.0) {
                    const double pClamped = std::max(epsilon, std::min(1.0 - epsilon, pRow[c]));
                    totalLoss -= tRow[c] * std::log(pClamped);
                }
            }
        }
        return totalLoss;
    }

    void derivativeBatch(const Matrix& predictions, const Matrix& targets, Matrix& dL_da) const override {
        constexpr double epsilon = 1e-15;
        dL_da.reallocate(predictions.rows(), predictions.cols());
        const std::size_t total = predictions.size();
        const double* p = predictions.dataPtr();
        const double* t = targets.dataPtr();
        double* out = dL_da.dataPtr();

        for (std::size_t i = 0; i < total; ++i) {
            const double pClamped = std::max(epsilon, std::min(1.0 - epsilon, p[i]));
            out[i] = -(t[i] / pClamped);
        }
    }
};

} // namespace miniann