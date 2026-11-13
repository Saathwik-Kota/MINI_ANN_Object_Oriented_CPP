#pragma once
#include <cstddef>
#include <memory>
#include <random>
#include <vector>
#include "MathVector.hpp"
#include "activations.hpp"
#include "optimizer.hpp"

namespace miniann {

class Neuron {
private:
    MathVector weights;
    double bias;
    MathVector velocity;
    double bias_velocity;
    std::shared_ptr<const IActivation> activationFunction;

    MathVector last_inputs;
    double last_z;
    double last_a;
    double delta;

public:
    Neuron(int numInputs,
           int layerSize,
           std::shared_ptr<const IActivation> activation,
           std::mt19937& rng);

    double forward(const MathVector& inputs);
    double evaluate(const MathVector& inputs) const;

    void computeOutputDelta(double dL_da);
    void computeHiddenDelta(double downstreamErrorSum);
    void updateParameters(IOptimizer& optimizer);
    void setParameters(double newBias, const std::vector<double>& newWeights);

    const MathVector& getWeights() const noexcept { return weights; }
    double getBias() const noexcept { return bias; }
    const MathVector& getLastInputs() const noexcept { return last_inputs; }
    double getLastZ() const noexcept { return last_z; }
    double getLastA() const noexcept { return last_a; }
    double getDelta() const noexcept { return delta; }
    std::size_t inputCount() const noexcept { return weights.size(); }
};

} // namespace miniann
