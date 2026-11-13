#pragma once
#include <cmath>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>
#include "MathVector.hpp"
#include "Matrix.hpp"

namespace miniann {

class IOptimizer {
protected:
    double learningRate;
    double weightDecay{0.0}; // L2 regularization (weight decay)
    std::size_t timestep_{0};

public:
    explicit IOptimizer(double lr, double wd = 0.0) : learningRate(lr), weightDecay(wd) {}
    virtual ~IOptimizer() = default;

    double getLearningRate() const noexcept { return learningRate; }
    void setLearningRate(double lr) noexcept { learningRate = lr; }

    double getWeightDecay() const noexcept { return weightDecay; }
    void setWeightDecay(double wd) noexcept { weightDecay = wd; }

    std::size_t getTimestep() const noexcept { return timestep_; }
    virtual void step() { ++timestep_; }
    virtual void reset() { timestep_ = 0; }

    virtual MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) = 0;
    virtual double calculateUpdate(double gradient, double& neuron_bias_velocity) = 0;

    virtual void update(Matrix& weights, const Matrix& grad_w, Matrix& velocity_w) = 0;
    virtual void update(std::vector<double>& biases, const std::vector<double>& grad_b, std::vector<double>& velocity_b) = 0;
};

class SGD : public IOptimizer {
public:
    explicit SGD(double lr, double wd = 0.0) : IOptimizer(lr, wd) {}

    MathVector calculateUpdate(const MathVector& gradients, MathVector& /*neuron_velocity*/) override {
        return gradients * learningRate;
    }
    double calculateUpdate(double gradient, double& /*neuron_bias_velocity*/) override {
        return gradient * learningRate;
    }

    void update(Matrix& weights, const Matrix& grad_w, Matrix& /*velocity_w*/) override {
        const std::size_t n = weights.size();
        double* wPtr = weights.dataPtr();
        const double* gPtr = grad_w.dataPtr();
        const double lr = learningRate;
        const double wd = weightDecay;

        for (std::size_t i = 0; i < n; ++i) {
            const double g = gPtr[i] + (wd * wPtr[i]);
            wPtr[i] -= lr * g;
        }
    }

    void update(std::vector<double>& biases, const std::vector<double>& grad_b, std::vector<double>& /*velocity_b*/) override {
        const std::size_t n = biases.size();
        const double lr = learningRate;
        for (std::size_t i = 0; i < n; ++i) {
            biases[i] -= lr * grad_b[i];
        }
    }
};

class Momentum : public IOptimizer {
private:
    double beta;
public:
    Momentum(double lr, double b = 0.9, double wd = 0.0) : IOptimizer(lr, wd), beta(b) {}
    double getBeta() const noexcept { return beta; }

    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) override {
        neuron_velocity = (neuron_velocity * beta) + (gradients * learningRate);
        return neuron_velocity;
    }
    double calculateUpdate(double gradient, double& neuron_bias_velocity) override {
        neuron_bias_velocity = (neuron_bias_velocity * beta) + (gradient * learningRate);
        return neuron_bias_velocity;
    }

    void update(Matrix& weights, const Matrix& grad_w, Matrix& velocity_w) override {
        const std::size_t n = weights.size();
        double* wPtr = weights.dataPtr();
        const double* gPtr = grad_w.dataPtr();
        double* vPtr = velocity_w.dataPtr();
        const double lr = learningRate;
        const double b = beta;
        const double wd = weightDecay;

        for (std::size_t i = 0; i < n; ++i) {
            const double g = gPtr[i] + (wd * wPtr[i]);
            vPtr[i] = (b * vPtr[i]) + (lr * g);
            wPtr[i] -= vPtr[i];
        }
    }

    void update(std::vector<double>& biases, const std::vector<double>& grad_b, std::vector<double>& velocity_b) override {
        const std::size_t n = biases.size();
        const double lr = learningRate;
        const double b = beta;
        for (std::size_t i = 0; i < n; ++i) {
            velocity_b[i] = (b * velocity_b[i]) + (lr * grad_b[i]);
            biases[i] -= velocity_b[i];
        }
    }
};

class Adam : public IOptimizer {
private:
    double beta1;
    double beta2;
    double epsilon;

    struct MatrixMomentState {
        Matrix m;
        Matrix v;
    };

    struct VectorMomentState {
        std::vector<double> m;
        std::vector<double> v;
    };

    std::unordered_map<const void*, MatrixMomentState> weightStates_;
    std::unordered_map<const void*, VectorMomentState> biasStates_;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.0)
        : IOptimizer(lr, wd), beta1(b1), beta2(b2), epsilon(eps) {}

    double getBeta1() const noexcept { return beta1; }
    double getBeta2() const noexcept { return beta2; }
    double getEpsilon() const noexcept { return epsilon; }

    void reset() override {
        IOptimizer::reset();
        weightStates_.clear();
        biasStates_.clear();
    }

    MathVector calculateUpdate(const MathVector& gradients, MathVector& /*neuron_velocity*/) override {
        return gradients * learningRate;
    }
    double calculateUpdate(double gradient, double& /*neuron_bias_velocity*/) override {
        return gradient * learningRate;
    }

    void update(Matrix& weights, const Matrix& grad_w, Matrix& /*velocity_w*/) override {
        const std::size_t n = weights.size();
        const void* key = weights.dataPtr();

        auto it = weightStates_.find(key);
        if (it == weightStates_.end()) {
            Matrix m(weights.rows(), weights.cols(), 0.0);
            Matrix v(weights.rows(), weights.cols(), 0.0);
            it = weightStates_.emplace(key, MatrixMomentState{std::move(m), std::move(v)}).first;
        }

        Matrix& m = it->second.m;
        Matrix& v = it->second.v;

        const double t = static_cast<double>(timestep_ == 0 ? 1 : timestep_);
        const double b1_corr = 1.0 - std::pow(beta1, t);
        const double b2_corr = 1.0 - std::pow(beta2, t);
        const double alpha_t = learningRate * (std::sqrt(b2_corr) / (b1_corr > 0.0 ? b1_corr : 1e-15));

        double* wPtr = weights.dataPtr();
        const double* gPtr = grad_w.dataPtr();
        double* mPtr = m.dataPtr();
        double* vPtr = v.dataPtr();

        const double wd = weightDecay;
        const double b1 = beta1;
        const double b2 = beta2;
        const double eps = epsilon;

        for (std::size_t i = 0; i < n; ++i) {
            const double g = gPtr[i] + (wd * wPtr[i]);
            mPtr[i] = (b1 * mPtr[i]) + ((1.0 - b1) * g);
            vPtr[i] = (b2 * vPtr[i]) + ((1.0 - b2) * g * g);

            const double stepVal = alpha_t * mPtr[i] / (std::sqrt(vPtr[i]) + eps);
            wPtr[i] -= stepVal;
        }
    }

    void update(std::vector<double>& biases, const std::vector<double>& grad_b, std::vector<double>& /*velocity_b*/) override {
        const std::size_t n = biases.size();
        const void* key = biases.data();

        auto it = biasStates_.find(key);
        if (it == biasStates_.end()) {
            std::vector<double> m(n, 0.0);
            std::vector<double> v(n, 0.0);
            it = biasStates_.emplace(key, VectorMomentState{std::move(m), std::move(v)}).first;
        }

        std::vector<double>& m = it->second.m;
        std::vector<double>& v = it->second.v;

        const double t = static_cast<double>(timestep_ == 0 ? 1 : timestep_);
        const double b1_corr = 1.0 - std::pow(beta1, t);
        const double b2_corr = 1.0 - std::pow(beta2, t);
        const double alpha_t = learningRate * (std::sqrt(b2_corr) / (b1_corr > 0.0 ? b1_corr : 1e-15));

        const double b1 = beta1;
        const double b2 = beta2;
        const double eps = epsilon;

        for (std::size_t i = 0; i < n; ++i) {
            const double g = grad_b[i];
            m[i] = (b1 * m[i]) + ((1.0 - b1) * g);
            v[i] = (b2 * v[i]) + ((1.0 - b2) * g * g);

            const double stepVal = alpha_t * m[i] / (std::sqrt(v[i]) + eps);
            biases[i] -= stepVal;
        }
    }
};

class RMSprop : public IOptimizer {
private:
    double alpha;
    double epsilon;
    std::unordered_map<const void*, Matrix> weightSqGrads_;
    std::unordered_map<const void*, std::vector<double>> biasSqGrads_;

public:
    RMSprop(double lr = 0.01, double a = 0.99, double eps = 1e-8, double wd = 0.0)
        : IOptimizer(lr, wd), alpha(a), epsilon(eps) {}

    void reset() override {
        IOptimizer::reset();
        weightSqGrads_.clear();
        biasSqGrads_.clear();
    }

    MathVector calculateUpdate(const MathVector& gradients, MathVector& /*neuron_velocity*/) override {
        return gradients * learningRate;
    }
    double calculateUpdate(double gradient, double& /*neuron_bias_velocity*/) override {
        return gradient * learningRate;
    }

    void update(Matrix& weights, const Matrix& grad_w, Matrix& /*velocity_w*/) override {
        const std::size_t n = weights.size();
        const void* key = weights.dataPtr();
        auto it = weightSqGrads_.find(key);
        if (it == weightSqGrads_.end()) {
            it = weightSqGrads_.emplace(key, Matrix(weights.rows(), weights.cols(), 0.0)).first;
        }

        Matrix& sq = it->second;
        double* wPtr = weights.dataPtr();
        const double* gPtr = grad_w.dataPtr();
        double* sqPtr = sq.dataPtr();

        const double lr = learningRate;
        const double a = alpha;
        const double eps = epsilon;
        const double wd = weightDecay;

        for (std::size_t i = 0; i < n; ++i) {
            const double g = gPtr[i] + (wd * wPtr[i]);
            sqPtr[i] = (a * sqPtr[i]) + ((1.0 - a) * g * g);
            wPtr[i] -= (lr * g) / (std::sqrt(sqPtr[i]) + eps);
        }
    }

    void update(std::vector<double>& biases, const std::vector<double>& grad_b, std::vector<double>& /*velocity_b*/) override {
        const std::size_t n = biases.size();
        const void* key = biases.data();
        auto it = biasSqGrads_.find(key);
        if (it == biasSqGrads_.end()) {
            it = biasSqGrads_.emplace(key, std::vector<double>(n, 0.0)).first;
        }

        std::vector<double>& sq = it->second;
        const double lr = learningRate;
        const double a = alpha;
        const double eps = epsilon;

        for (std::size_t i = 0; i < n; ++i) {
            const double g = grad_b[i];
            sq[i] = (a * sq[i]) + ((1.0 - a) * g * g);
            biases[i] -= (lr * g) / (std::sqrt(sq[i]) + eps);
        }
    }
};

} // namespace miniann