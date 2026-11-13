#pragma once
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "Layer.hpp"
#include "MathVector.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

namespace miniann {

struct TrainingConfig {
    int epochs{100};
    std::size_t batchSize{32};
    double lrDecayRate{1.0};
    int lrDecayStep{0};
    int patience{0}; // Early stopping patience (0 = disabled)
    double minDelta{1e-5};
    bool verbose{true};
    int logInterval{100};
};

struct TrainingHistory {
    std::vector<double> trainLoss;
    std::vector<double> valLoss;
    int stoppedEpoch{0};
    bool earlyStopped{false};
};

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    std::unique_ptr<IOptimizer> optimizer;
    std::unique_ptr<ILossFunction> lossFunction;
    std::mt19937 rng;

    mutable Matrix eval_scratch_;
    Matrix train_scratch_;

    void forwardPassBatch(const Matrix& inputs, Matrix& outputs);

public:
    explicit NeuralNetwork(unsigned int seed = std::random_device{}());
    NeuralNetwork(std::unique_ptr<IOptimizer> opt,
                  std::unique_ptr<ILossFunction> loss,
                  unsigned int seed = std::random_device{}());
    ~NeuralNetwork() = default;

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) noexcept = default;
    NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default;

    void addLayer(int numNeurons,
                  int numInputsPerNeuron,
                  std::shared_ptr<const IActivation> activation,
                  double dropoutRate = 0.0);

    void setTrainingMode(bool isTraining) noexcept;

    // Single-sample inference
    MathVector predict(const MathVector& inputs) const;
    std::vector<double> predict(const std::vector<double>& inputs) const;

    // High-performance batch inference
    void predict(const Matrix& inputs, Matrix& outputs) const;
    Matrix predict(const Matrix& inputs) const;

    // Loss calculation
    double computeLoss(const MathVector& prediction, const MathVector& target) const;
    double computeLoss(const Matrix& predictions, const Matrix& targets) const;

    // Sample and mini-batch forward/backward step
    double forwardBackwardSample(const MathVector& input, const MathVector& target);
    double forwardBackwardBatch(const Matrix& batchX, const Matrix& batchY);

    // Advanced Training with validation, early stopping, and lr decay
    TrainingHistory train(const Matrix& X_train,
                          const Matrix& y_train,
                          const Matrix& X_val,
                          const Matrix& y_val,
                          const TrainingConfig& config);

    // Convenience training overloads
    void train(const Matrix& X,
               const Matrix& y,
               int epochs,
               std::size_t batchSize = 32);

    void train(const std::vector<MathVector>& X,
               const std::vector<MathVector>& y,
               int epochs,
               std::size_t batchSize = 1);

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs,
               std::size_t batchSize = 1);

    // Serialization
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);

    // Topology and component queries
    std::size_t layerCount() const noexcept { return layers.size(); }
    const std::vector<Layer>& getLayers() const noexcept { return layers; }
    Layer& layerAt(std::size_t index) { return layers.at(index); }
    const Layer& layerAt(std::size_t index) const { return layers.at(index); }

    IOptimizer* getOptimizer() noexcept { return optimizer.get(); }
    const IOptimizer* getOptimizer() const noexcept { return optimizer.get(); }
    ILossFunction* getLossFunction() noexcept { return lossFunction.get(); }
    const ILossFunction* getLossFunction() const noexcept { return lossFunction.get(); }
};

} // namespace miniann
