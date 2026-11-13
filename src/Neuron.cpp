#include "Neuron.hpp"
#include <stdexcept>
#include <utility>

namespace miniann {

Neuron::Neuron(int numInputs,
               int layerSize,
               std::shared_ptr<const IActivation> activation,
               std::mt19937& rng)
    : weights(numInputs, 0.0),
      bias(0.0),
      velocity(numInputs, 0.0),
      bias_velocity(0.0),
      activationFunction(std::move(activation)),
      last_inputs(numInputs, 0.0),
      last_z(0.0),
      last_a(0.0),
      delta(0.0) {
    if (numInputs <= 0 || layerSize <= 0) {
        throw std::invalid_argument("Neuron input count and layer size must be positive.");
    }
    if (!activationFunction) {
        throw std::invalid_argument("Neuron activation function cannot be null.");
    }

    double stddev = activationFunction->weightInitStdDev(numInputs, layerSize);
    std::normal_distribution<double> dis(0.0, stddev);

    for (int i = 0; i < numInputs; ++i) {
        weights.data[static_cast<std::size_t>(i)] = dis(rng);
    }
}

double Neuron::forward(const MathVector& inputs) {
    this->last_inputs = inputs;
    this->last_z = (this->weights * inputs) + this->bias;
    this->last_a = this->activationFunction->activate(this->last_z);
    return this->last_a;
}

double Neuron::evaluate(const MathVector& inputs) const {
    const double z = (this->weights * inputs) + this->bias;
    return this->activationFunction->activate(z);
}

void Neuron::computeOutputDelta(double dL_da) {
    const double da_dz = this->activationFunction->derivative(this->last_z);
    this->delta = dL_da * da_dz;
}

void Neuron::computeHiddenDelta(double downstreamErrorSum) {
    const double da_dz = this->activationFunction->derivative(this->last_z);
    this->delta = downstreamErrorSum * da_dz;
}

void Neuron::updateParameters(IOptimizer& optimizer) {
    MathVector gradients = this->last_inputs * this->delta;
    MathVector weight_update = optimizer.calculateUpdate(gradients, this->velocity);
    double bias_update = optimizer.calculateUpdate(this->delta, this->bias_velocity);

    this->weights = this->weights - weight_update;
    this->bias -= bias_update;
}

void Neuron::setParameters(double newBias, const std::vector<double>& newWeights) {
    if (newWeights.size() != this->weights.size()) {
        throw std::invalid_argument("Weight count mismatch in Neuron::setParameters.");
    }
    this->bias = newBias;
    this->weights.data = newWeights;
}

} // namespace miniann
