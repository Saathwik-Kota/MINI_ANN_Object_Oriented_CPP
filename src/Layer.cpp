#include "Layer.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace miniann {

// ======================== NeuronProxy Implementation ========================

double NeuronProxy::getBias() const {
    return layer_->getBias(index_);
}

MathVector NeuronProxy::getWeights() const {
    std::vector<double> w(layer_->inputSize());
    layer_->getWeights().extractRow(index_, w);
    return MathVector(w);
}

double NeuronProxy::getDelta() const {
    if (layer_->getDelta().empty()) return 0.0;
    return layer_->getDelta()(0, index_);
}

MathVector NeuronProxy::getLastInputs() const {
    if (layer_->getLastInputs().empty()) return MathVector(layer_->inputSize(), 0.0);
    std::vector<double> in(layer_->inputSize());
    layer_->getLastInputs().extractRow(0, in);
    return MathVector(in);
}

double NeuronProxy::getLastZ() const {
    if (layer_->getLastZ().empty()) return 0.0;
    return layer_->getLastZ()(0, index_);
}

double NeuronProxy::getLastA() const {
    if (layer_->getLastA().empty()) return 0.0;
    return layer_->getLastA()(0, index_);
}

std::size_t NeuronProxy::inputCount() const {
    return static_cast<std::size_t>(layer_->inputSize());
}

void NeuronProxy::setParameters(double newBias, const std::vector<double>& newWeights) {
    layer_->setNeuronParameters(index_, newBias, newWeights);
}

// ===================== ConstNeuronProxy Implementation =====================

double ConstNeuronProxy::getBias() const {
    return layer_->getBias(index_);
}

MathVector ConstNeuronProxy::getWeights() const {
    std::vector<double> w(layer_->inputSize());
    layer_->getWeights().extractRow(index_, w);
    return MathVector(w);
}

double ConstNeuronProxy::getDelta() const {
    if (layer_->getDelta().empty()) return 0.0;
    return layer_->getDelta()(0, index_);
}

MathVector ConstNeuronProxy::getLastInputs() const {
    if (layer_->getLastInputs().empty()) return MathVector(layer_->inputSize(), 0.0);
    std::vector<double> in(layer_->inputSize());
    layer_->getLastInputs().extractRow(0, in);
    return MathVector(in);
}

double ConstNeuronProxy::getLastZ() const {
    if (layer_->getLastZ().empty()) return 0.0;
    return layer_->getLastZ()(0, index_);
}

double ConstNeuronProxy::getLastA() const {
    if (layer_->getLastA().empty()) return 0.0;
    return layer_->getLastA()(0, index_);
}

std::size_t ConstNeuronProxy::inputCount() const {
    return static_cast<std::size_t>(layer_->inputSize());
}

// ======================== Layer Implementation =============================

Layer::Layer(int numNeurons,
             int numInputsPerNeuron,
             std::shared_ptr<const IActivation> activation,
             std::mt19937& rng,
             double dropoutRate)
    : numNeurons_(numNeurons),
      numInputs_(numInputsPerNeuron),
      activationFunction_(std::move(activation)),
      weights_(static_cast<std::size_t>(numNeurons), static_cast<std::size_t>(numInputsPerNeuron), 0.0),
      biases_(static_cast<std::size_t>(numNeurons), 0.0),
      velocity_w_(static_cast<std::size_t>(numNeurons), static_cast<std::size_t>(numInputsPerNeuron), 0.0),
      velocity_b_(static_cast<std::size_t>(numNeurons), 0.0),
      dropoutRate_(dropoutRate) {
    if (numNeurons <= 0 || numInputsPerNeuron <= 0) {
        throw std::invalid_argument("Layer neuron count and input count must be positive.");
    }
    if (!activationFunction_) {
        throw std::invalid_argument("Layer activation function cannot be null.");
    }
    if (dropoutRate_ < 0.0 || dropoutRate_ >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in [0.0, 1.0).");
    }

    const double stddev = activationFunction_->weightInitStdDev(numInputsPerNeuron, numNeurons);
    std::normal_distribution<double> dis(0.0, stddev);

    double* wPtr = weights_.dataPtr();
    const std::size_t totalWeights = weights_.size();
    for (std::size_t i = 0; i < totalWeights; ++i) {
        wPtr[i] = dis(rng);
    }
}

void Layer::forward(const Matrix& inputs, Matrix& outputs) {
    const std::size_t batchSize = inputs.rows();
    if (inputs.cols() != static_cast<std::size_t>(numInputs_)) {
        throw std::invalid_argument("Input feature dimension mismatch in Layer::forward.");
    }

    last_inputs_ = inputs; // cache input for gradient computation
    last_z_.reallocate(batchSize, static_cast<std::size_t>(numNeurons_));
    last_a_.reallocate(batchSize, static_cast<std::size_t>(numNeurons_));

    // Z = inputs * weights_^T
    Matrix::gemm(inputs, weights_, last_z_, false, true, 1.0, 0.0);
    last_z_.addBiasRow(biases_);

    // A = activation(Z)
    activationFunction_->apply(last_z_, last_a_);

    // Inverted Dropout
    if (trainingMode_ && dropoutRate_ > 0.0) {
        dropoutMask_.reallocate(batchSize, static_cast<std::size_t>(numNeurons_));
        const double keepProb = 1.0 - dropoutRate_;
        const double scale = 1.0 / keepProb;
        double* aPtr = last_a_.dataPtr();
        double* mPtr = dropoutMask_.dataPtr();
        const std::size_t total = last_a_.size();

        // Thread-safe / static distribution
        static thread_local std::mt19937 maskRng(777);
        std::bernoulli_distribution bern(keepProb);

        for (std::size_t i = 0; i < total; ++i) {
            const double maskVal = bern(maskRng) ? scale : 0.0;
            mPtr[i] = maskVal;
            aPtr[i] *= maskVal;
        }
    }

    outputs = last_a_;
}

void Layer::evaluate(const Matrix& inputs, Matrix& outputs) const {
    const std::size_t batchSize = inputs.rows();
    if (inputs.cols() != static_cast<std::size_t>(numInputs_)) {
        throw std::invalid_argument("Input feature dimension mismatch in Layer::evaluate.");
    }

    outputs.reallocate(batchSize, static_cast<std::size_t>(numNeurons_));
    Matrix::gemm(inputs, weights_, outputs, false, true, 1.0, 0.0);
    outputs.addBiasRow(biases_);

    activationFunction_->apply(outputs, outputs);
}

MathVector Layer::forward(const MathVector& inputs) {
    Matrix inMat(1, inputs.size(), inputs.data);
    Matrix outMat;
    forward(inMat, outMat);
    return MathVector(outMat.data());
}

MathVector Layer::evaluate(const MathVector& inputs) const {
    Matrix inMat(1, inputs.size(), inputs.data);
    Matrix outMat;
    evaluate(inMat, outMat);
    return MathVector(outMat.data());
}

void Layer::computeOutputDeltas(const Matrix& targets, const ILossFunction& lossFunction) {
    const std::size_t batchSize = last_a_.rows();
    const std::size_t outDim = last_a_.cols();
    if (targets.rows() != batchSize || targets.cols() != outDim) {
        throw std::invalid_argument("Target dimension mismatch in computeOutputDeltas.");
    }

    delta_.reallocate(batchSize, outDim);

    // Fast-path: Softmax + CategoricalCrossEntropy combined gradient is exactly (A - Y)
    if (activationFunction_->name() == "Softmax" && lossFunction.name() == "CategoricalCrossEntropy") {
        const double* aPtr = last_a_.dataPtr();
        const double* tPtr = targets.dataPtr();
        double* deltaPtr = delta_.dataPtr();
        const std::size_t total = batchSize * outDim;
        for (std::size_t i = 0; i < total; ++i) {
            deltaPtr[i] = aPtr[i] - tPtr[i];
        }
    } else {
        lossFunction.derivativeBatch(last_a_, targets, loss_grad_buf_);
        activationFunction_->applyDerivative(last_z_, last_a_, loss_grad_buf_, delta_);
    }

    // Apply dropout mask to gradient if active
    if (trainingMode_ && dropoutRate_ > 0.0 && !dropoutMask_.empty()) {
        const std::size_t total = delta_.size();
        double* dPtr = delta_.dataPtr();
        const double* mPtr = dropoutMask_.dataPtr();
        for (std::size_t i = 0; i < total; ++i) {
            dPtr[i] *= mPtr[i];
        }
    }
}

void Layer::computeHiddenDeltas(const Layer& nextLayer) {
    const std::size_t batchSize = last_a_.rows();
    const std::size_t hiddenDim = last_a_.cols();

    // error = delta_next * weights_next  (shape: batchSize x hiddenDim)
    downstream_error_.reallocate(batchSize, hiddenDim);
    Matrix::gemm(nextLayer.getDelta(), nextLayer.getWeights(), downstream_error_, false, false, 1.0, 0.0);

    delta_.reallocate(batchSize, hiddenDim);
    activationFunction_->applyDerivative(last_z_, last_a_, downstream_error_, delta_);

    // Apply dropout mask to gradient if active
    if (trainingMode_ && dropoutRate_ > 0.0 && !dropoutMask_.empty()) {
        const std::size_t total = delta_.size();
        double* dPtr = delta_.dataPtr();
        const double* mPtr = dropoutMask_.dataPtr();
        for (std::size_t i = 0; i < total; ++i) {
            dPtr[i] *= mPtr[i];
        }
    }
}

void Layer::computeGradients() {
    const std::size_t batchSize = delta_.rows();
    if (batchSize == 0) return;

    // grad_w = (1 / B) * delta_^T * last_inputs_
    // delta_ is (B x numNeurons), last_inputs_ is (B x numInputs)
    // grad_w_ is (numNeurons x numInputs)
    const double scale = 1.0 / static_cast<double>(batchSize);
    Matrix::gemm(delta_, last_inputs_, grad_w_, true, false, scale, 0.0);

    // grad_b = (1 / B) * sum_rows(delta_)
    delta_.sumColumns(grad_b_);
    for (double& b : grad_b_) {
        b *= scale;
    }
}

void Layer::updateParameters(IOptimizer& optimizer) {
    computeGradients();
    optimizer.update(weights_, grad_w_, velocity_w_);
    optimizer.update(biases_, grad_b_, velocity_b_);
}

double Layer::getBias(std::size_t neuronIdx) const {
    return biases_.at(neuronIdx);
}

void Layer::setBias(std::size_t neuronIdx, double b) {
    biases_.at(neuronIdx) = b;
}

void Layer::setNeuronParameters(std::size_t neuronIdx, double bias, const std::vector<double>& weights) {
    if (neuronIdx >= static_cast<std::size_t>(numNeurons_)) {
        throw std::out_of_range("Neuron index out of range.");
    }
    if (weights.size() != static_cast<std::size_t>(numInputs_)) {
        throw std::invalid_argument("Weight count mismatch in Layer::setNeuronParameters.");
    }
    biases_[neuronIdx] = bias;
    weights_.setRow(neuronIdx, weights.data());
}

} // namespace miniann
