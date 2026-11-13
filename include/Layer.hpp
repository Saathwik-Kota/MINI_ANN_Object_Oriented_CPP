#pragma once
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "MathVector.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

namespace miniann {

class Layer;

class NeuronProxy {
private:
    Layer* layer_{nullptr};
    std::size_t index_{0};

public:
    NeuronProxy(Layer* layer, std::size_t index) : layer_(layer), index_(index) {}

    double getBias() const;
    MathVector getWeights() const;
    double getDelta() const;
    MathVector getLastInputs() const;
    double getLastZ() const;
    double getLastA() const;
    std::size_t inputCount() const;
    void setParameters(double newBias, const std::vector<double>& newWeights);
};

class ConstNeuronProxy {
private:
    const Layer* layer_{nullptr};
    std::size_t index_{0};

public:
    ConstNeuronProxy(const Layer* layer, std::size_t index) : layer_(layer), index_(index) {}

    double getBias() const;
    MathVector getWeights() const;
    double getDelta() const;
    MathVector getLastInputs() const;
    double getLastZ() const;
    double getLastA() const;
    std::size_t inputCount() const;
};

class Layer {
private:
    int numNeurons_{0};
    int numInputs_{0};
    std::shared_ptr<const IActivation> activationFunction_;

    // Contiguous parameters: weights_ is (numNeurons x numInputs)
    Matrix weights_;
    std::vector<double> biases_;
    Matrix velocity_w_;
    std::vector<double> velocity_b_;

    // Regularization & Dropout
    double dropoutRate_{0.0};
    bool trainingMode_{true};
    Matrix dropoutMask_;

    // Pre-allocated forward/backward scratch buffers (zero inner-loop allocations)
    Matrix last_inputs_;
    Matrix last_z_;
    Matrix last_a_;
    Matrix delta_;
    Matrix grad_w_;
    std::vector<double> grad_b_;
    Matrix downstream_error_;
    Matrix loss_grad_buf_;

public:
    Layer(int numNeurons,
          int numInputsPerNeuron,
          std::shared_ptr<const IActivation> activation,
          std::mt19937& rng,
          double dropoutRate = 0.0);

    void setTrainingMode(bool isTraining) noexcept { trainingMode_ = isTraining; }
    bool isTrainingMode() const noexcept { return trainingMode_; }
    void setDropoutRate(double rate) {
        if (rate < 0.0 || rate >= 1.0) {
            throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
        }
        dropoutRate_ = rate;
    }
    double getDropoutRate() const noexcept { return dropoutRate_; }

    // High-performance batch forward & evaluate
    void forward(const Matrix& inputs, Matrix& outputs);
    void evaluate(const Matrix& inputs, Matrix& outputs) const;

    // MathVector convenience adapters (batch size = 1)
    MathVector forward(const MathVector& inputs);
    MathVector evaluate(const MathVector& inputs) const;

    // Backpropagation
    void computeOutputDeltas(const Matrix& targets, const ILossFunction& lossFunction);
    void computeHiddenDeltas(const Layer& nextLayer);
    void computeGradients();
    void updateParameters(IOptimizer& optimizer);

    // Diagnostics & Inspection
    std::size_t size() const noexcept { return static_cast<std::size_t>(numNeurons_); }
    int inputSize() const noexcept { return numInputs_; }
    std::string activationName() const { return activationFunction_->name(); }
    const std::shared_ptr<const IActivation>& getActivation() const noexcept { return activationFunction_; }

    const Matrix& getWeights() const noexcept { return weights_; }
    Matrix& getWeights() noexcept { return weights_; }
    const std::vector<double>& getBiases() const noexcept { return biases_; }
    std::vector<double>& getBiases() noexcept { return biases_; }

    double getBias(std::size_t neuronIdx) const;
    void setBias(std::size_t neuronIdx, double b);
    void setNeuronParameters(std::size_t neuronIdx, double bias, const std::vector<double>& weights);

    const Matrix& getLastInputs() const noexcept { return last_inputs_; }
    const Matrix& getLastZ() const noexcept { return last_z_; }
    const Matrix& getLastA() const noexcept { return last_a_; }
    const Matrix& getDelta() const noexcept { return delta_; }
    const Matrix& getGradW() const noexcept { return grad_w_; }
    const std::vector<double>& getGradB() const noexcept { return grad_b_; }

    NeuronProxy neuronAt(std::size_t index) { return NeuronProxy(this, index); }
    ConstNeuronProxy neuronAt(std::size_t index) const { return ConstNeuronProxy(this, index); }
};

} // namespace miniann
