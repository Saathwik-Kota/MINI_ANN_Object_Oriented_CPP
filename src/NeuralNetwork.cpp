#include "NeuralNetwork.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace miniann {

namespace {

std::string trimCR(std::string s) {
  if (!s.empty() && s.back() == '\r') {
    s.pop_back();
  }
  return s;
}

std::vector<std::string> splitCSVLine(const std::string &line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    tokens.push_back(item);
  }
  if (!line.empty() && line.back() == ',') {
    tokens.emplace_back("");
  }
  return tokens;
}

int parseStrictInt(const std::string &token) {
  if (token.empty())
    throw std::invalid_argument("Empty integer field.");
  std::size_t pos = 0;
  int val = std::stoi(token, &pos);
  if (pos != token.size())
    throw std::invalid_argument("Invalid integer field: " + token);
  return val;
}

double parseStrictDouble(const std::string &token) {
  if (token.empty())
    throw std::invalid_argument("Empty floating-point field.");
  std::size_t pos = 0;
  double val = std::stod(token, &pos);
  if (pos != token.size() || !std::isfinite(val)) {
    throw std::invalid_argument("Invalid floating-point field: " + token);
  }
  return val;
}

} // namespace

NeuralNetwork::NeuralNetwork(unsigned int seed) : rng(seed) {}

NeuralNetwork::NeuralNetwork(std::unique_ptr<IOptimizer> opt,
                             std::unique_ptr<ILossFunction> loss,
                             unsigned int seed)
    : optimizer(std::move(opt)), lossFunction(std::move(loss)), rng(seed) {}

void NeuralNetwork::addLayer(int numNeurons, int numInputsPerNeuron,
                             std::shared_ptr<const IActivation> activation,
                             double dropoutRate) {
  if (!layers.empty()) {
    const int expectedInputs = static_cast<int>(layers.back().size());
    if (numInputsPerNeuron != expectedInputs) {
      throw std::invalid_argument(
          "Layer input size (" + std::to_string(numInputsPerNeuron) +
          ") does not match previous layer output size (" +
          std::to_string(expectedInputs) + ").");
    }
  }
  layers.emplace_back(numNeurons, numInputsPerNeuron, std::move(activation),
                      rng, dropoutRate);
}

void NeuralNetwork::setTrainingMode(bool isTraining) noexcept {
  for (auto &layer : layers) {
    layer.setTrainingMode(isTraining);
  }
}

void NeuralNetwork::forwardPassBatch(const Matrix &inputs, Matrix &outputs) {
  if (layers.empty()) {
    throw std::runtime_error(
        "Cannot run forward pass on an empty NeuralNetwork.");
  }
  if (inputs.cols() != static_cast<std::size_t>(layers.front().inputSize())) {
    throw std::invalid_argument(
        "Input dimension (" + std::to_string(inputs.cols()) +
        ") does not match network input size (" +
        std::to_string(layers.front().inputSize()) + ").");
  }

  const Matrix *currentInput = &inputs;
  Matrix intermediate;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    Matrix &targetBuffer = (i == layers.size() - 1)
                               ? outputs
                               : ((i % 2 == 0) ? intermediate : train_scratch_);
    layers[i].forward(*currentInput, targetBuffer);
    currentInput = &targetBuffer;
  }
}

void NeuralNetwork::predict(const Matrix &inputs, Matrix &outputs) const {
  if (layers.empty()) {
    throw std::runtime_error("Cannot run predict on an empty NeuralNetwork.");
  }
  if (inputs.cols() != static_cast<std::size_t>(layers.front().inputSize())) {
    throw std::invalid_argument(
        "Input dimension (" + std::to_string(inputs.cols()) +
        ") does not match network input size (" +
        std::to_string(layers.front().inputSize()) + ").");
  }

  const Matrix *currentInput = &inputs;
  Matrix intermediate;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    Matrix &targetBuffer = (i == layers.size() - 1)
                               ? outputs
                               : ((i % 2 == 0) ? intermediate : eval_scratch_);
    layers[i].evaluate(*currentInput, targetBuffer);
    currentInput = &targetBuffer;
  }
}

Matrix NeuralNetwork::predict(const Matrix &inputs) const {
  Matrix outputs;
  predict(inputs, outputs);
  return outputs;
}

MathVector NeuralNetwork::predict(const MathVector &inputs) const {
  Matrix inMat(1, inputs.size(), inputs.data);
  Matrix outMat;
  predict(inMat, outMat);
  return MathVector(outMat.data());
}

std::vector<double>
NeuralNetwork::predict(const std::vector<double> &inputs) const {
  Matrix inMat(1, inputs.size(), inputs);
  Matrix outMat;
  predict(inMat, outMat);
  return outMat.data();
}

double NeuralNetwork::computeLoss(const MathVector &prediction,
                                  const MathVector &target) const {
  if (!lossFunction) {
    throw std::runtime_error("Loss function is not set.");
  }
  if (prediction.size() != target.size()) {
    throw std::invalid_argument(
        "Prediction and target dimensions do not match.");
  }
  Matrix pMat(1, prediction.size(), prediction.data);
  Matrix tMat(1, target.size(), target.data);
  return lossFunction->calculateBatch(pMat, tMat);
}

double NeuralNetwork::computeLoss(const Matrix &predictions,
                                  const Matrix &targets) const {
  if (!lossFunction) {
    throw std::runtime_error("Loss function is not set.");
  }
  if (predictions.rows() != targets.rows() ||
      predictions.cols() != targets.cols()) {
    throw std::invalid_argument(
        "Predictions and targets shape mismatch in computeLoss.");
  }
  return lossFunction->calculateBatch(predictions, targets);
}

double NeuralNetwork::forwardBackwardBatch(const Matrix &batchX,
                                           const Matrix &batchY) {
  if (layers.empty()) {
    throw std::runtime_error(
        "Cannot run backpropagation on an empty NeuralNetwork.");
  }
  if (!lossFunction) {
    throw std::runtime_error(
        "Loss function must be set before backpropagation.");
  }

  if (optimizer) {
    optimizer->step();
  }

  Matrix output;
  forwardPassBatch(batchX, output);
  const double batchLoss = computeLoss(output, batchY);

  layers.back().computeOutputDeltas(batchY, *lossFunction);
  for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
    layers[static_cast<std::size_t>(l)].computeHiddenDeltas(
        layers[static_cast<std::size_t>(l + 1)]);
  }

  return batchLoss;
}

double NeuralNetwork::forwardBackwardSample(const MathVector &input,
                                            const MathVector &target) {
  Matrix inMat(1, input.size(), input.data);
  Matrix tgtMat(1, target.size(), target.data);
  return forwardBackwardBatch(inMat, tgtMat);
}

TrainingHistory NeuralNetwork::train(const Matrix &X_train,
                                     const Matrix &y_train, const Matrix &X_val,
                                     const Matrix &y_val,
                                     const TrainingConfig &config) {
  if (layers.empty()) {
    throw std::runtime_error("Cannot train an empty NeuralNetwork.");
  }
  if (!optimizer || !lossFunction) {
    throw std::runtime_error(
        "Optimizer and loss function must be set before training.");
  }
  if (X_train.empty() || X_train.rows() != y_train.rows()) {
    throw std::invalid_argument("Training features and targets must be "
                                "non-empty with equal row counts.");
  }
  if (config.epochs <= 0) {
    throw std::invalid_argument("Number of epochs must be positive.");
  }
  if (config.batchSize == 0) {
    throw std::invalid_argument("Batch size must be positive.");
  }

  const std::size_t numSamples = X_train.rows();
  const std::size_t inDim = X_train.cols();
  const std::size_t outDim = y_train.cols();

  if (inDim != static_cast<std::size_t>(layers.front().inputSize())) {
    throw std::invalid_argument(
        "Feature dimension mismatch with network input size.");
  }
  if (outDim != layers.back().size()) {
    throw std::invalid_argument(
        "Target dimension mismatch with network output size.");
  }

  const bool hasValidation = (!X_val.empty() && X_val.rows() == y_val.rows());

  std::vector<std::size_t> indices(numSamples);
  std::iota(indices.begin(), indices.end(), 0);

  const std::size_t maxBatch = std::min(config.batchSize, numSamples);
  Matrix batchX(maxBatch, inDim);
  Matrix batchY(maxBatch, outDim);

  TrainingHistory history;
  history.trainLoss.reserve(static_cast<std::size_t>(config.epochs));
  if (hasValidation)
    history.valLoss.reserve(static_cast<std::size_t>(config.epochs));

  setTrainingMode(true);

  double bestValLoss = std::numeric_limits<double>::infinity();
  int patienceCounter = 0;

  // Snapshot of best weights for early stopping
  std::vector<Matrix> bestWeights;
  std::vector<std::vector<double>> bestBiases;
  if (hasValidation && config.patience > 0) {
    for (const auto &l : layers) {
      bestWeights.push_back(l.getWeights());
      bestBiases.push_back(l.getBiases());
    }
  }

  for (int epoch = 0; epoch < config.epochs; ++epoch) {
    // Learning rate step decay
    if (config.lrDecayStep > 0 && epoch > 0 &&
        (epoch % config.lrDecayStep == 0)) {
      const double newLr = optimizer->getLearningRate() * config.lrDecayRate;
      optimizer->setLearningRate(newLr);
      if (config.verbose && (epoch % config.logInterval == 0)) {
        std::cout << "  [LR Scheduler] Adjusted learning rate to " << newLr
                  << "\n";
      }
    }

    std::shuffle(indices.begin(), indices.end(), rng);
    double total_loss = 0.0;

    for (std::size_t startIdx = 0; startIdx < numSamples;
         startIdx += config.batchSize) {
      const std::size_t currentBatchSize =
          std::min(config.batchSize, numSamples - startIdx);
      batchX.reallocate(currentBatchSize, inDim);
      batchY.reallocate(currentBatchSize, outDim);

      for (std::size_t b = 0; b < currentBatchSize; ++b) {
        const std::size_t sampleIdx = indices[startIdx + b];
        batchX.setRow(b, X_train.rowPtr(sampleIdx));
        batchY.setRow(b, y_train.rowPtr(sampleIdx));
      }

      const double bLoss = forwardBackwardBatch(batchX, batchY);
      total_loss += bLoss;

      for (auto &layer : layers) {
        layer.updateParameters(*optimizer);
      }
    }

    const double meanTrainLoss =
        total_loss / static_cast<double>(numSamples * outDim);
    history.trainLoss.push_back(meanTrainLoss);

    double meanValLoss = 0.0;
    if (hasValidation) {
      Matrix valPred;
      predict(X_val, valPred);
      meanValLoss = computeLoss(valPred, y_val) /
                    static_cast<double>(X_val.rows() * y_val.cols());
      history.valLoss.push_back(meanValLoss);
    }

    if (config.verbose &&
        ((epoch % config.logInterval == 0) || epoch == config.epochs - 1)) {
      std::cout << "Epoch " << std::setw(5) << epoch
                << " | Train Loss: " << std::fixed << std::setprecision(5)
                << meanTrainLoss;
      if (hasValidation) {
        std::cout << " | Val Loss: " << meanValLoss;
      }
      std::cout << "\n";
    }

    // Early stopping check
    if (hasValidation && config.patience > 0) {
      if (meanValLoss < (bestValLoss - config.minDelta)) {
        bestValLoss = meanValLoss;
        patienceCounter = 0;
        for (std::size_t l = 0; l < layers.size(); ++l) {
          bestWeights[l] = layers[l].getWeights();
          bestBiases[l] = layers[l].getBiases();
        }
      } else {
        ++patienceCounter;
        if (patienceCounter >= config.patience) {
          if (config.verbose) {
            std::cout << "Early stopping triggered at epoch " << epoch
                      << " (best validation loss: " << bestValLoss << ")\n";
          }
          // Restore best weights
          for (std::size_t l = 0; l < layers.size(); ++l) {
            layers[l].getWeights() = bestWeights[l];
            layers[l].getBiases() = bestBiases[l];
          }
          history.stoppedEpoch = epoch;
          history.earlyStopped = true;
          break;
        }
      }
    }
    history.stoppedEpoch = epoch;
  }

  setTrainingMode(false);
  return history;
}

void NeuralNetwork::train(const Matrix &X, const Matrix &y, int epochs,
                          std::size_t batchSize) {
  TrainingConfig config;
  config.epochs = epochs;
  config.batchSize = batchSize;
  config.logInterval = 500;
  config.verbose = true;
  train(X, y, Matrix(), Matrix(), config);
}

void NeuralNetwork::train(const std::vector<MathVector> &X,
                          const std::vector<MathVector> &y, int epochs,
                          std::size_t batchSize) {
  if (X.empty() || X.size() != y.size()) {
    throw std::invalid_argument("Sample counts must be non-empty and equal.");
  }

  const std::size_t numSamples = X.size();
  const std::size_t inDim = X[0].size();
  const std::size_t outDim = y[0].size();

  Matrix xMat(numSamples, inDim);
  Matrix yMat(numSamples, outDim);

  for (std::size_t i = 0; i < numSamples; ++i) {
    xMat.setRow(i, X[i].data.data());
    yMat.setRow(i, y[i].data.data());
  }

  train(xMat, yMat, epochs, batchSize);
}

void NeuralNetwork::train(const std::vector<std::vector<double>> &X_raw,
                          const std::vector<std::vector<double>> &y_raw,
                          int epochs, std::size_t batchSize) {
  if (X_raw.empty() || X_raw.size() != y_raw.size()) {
    throw std::invalid_argument("Sample counts must be non-empty and equal.");
  }

  Matrix xMat = Matrix::from2D(X_raw);
  Matrix yMat = Matrix::from2D(y_raw);
  train(xMat, yMat, epochs, batchSize);
}

bool NeuralNetwork::saveModel(const std::string &filename) const {
  if (layers.empty()) {
    std::cerr << "Error: Cannot save an empty model.\n";
    return false;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for saving model: " << filename
              << "\n";
    return false;
  }

  file << std::setprecision(17);
  file << "MINIANN_V1\n";
  file << layers.size() << "\n";

  for (std::size_t l = 0; l < layers.size(); ++l) {
    const Layer &layer = layers[l];
    file << l << "," << layer.size() << "," << layer.inputSize() << ","
         << layer.activationName() << "\n";
  }

  for (std::size_t l = 0; l < layers.size(); ++l) {
    for (std::size_t n = 0; n < layers[l].size(); ++n) {
      file << l << "," << n << "," << layers[l].getBias(n);
      const double *w = layers[l].getWeights().rowPtr(n);
      for (int i = 0; i < layers[l].inputSize(); ++i) {
        file << "," << w[i];
      }
      file << "\n";
    }
  }

  file.flush();
  if (!file.good()) {
    std::cerr << "Error: Failed while writing model to " << filename << "\n";
    return false;
  }

  file.close();
  std::cout << "Model saved to " << filename << "\n";
  return true;
}

bool NeuralNetwork::loadModel(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open model file: " << filename << "\n";
    return false;
  }

  try {
    std::string line;
    if (!std::getline(file, line)) {
      throw std::runtime_error("Model file is empty.");
    }
    line = trimCR(line);
    if (line != "MINIANN_V1") {
      throw std::runtime_error(
          "Unsupported or missing model header (expected MINIANN_V1).");
    }

    if (!std::getline(file, line)) {
      throw std::runtime_error("Missing layer count in model file.");
    }
    const int numLayers = parseStrictInt(trimCR(line));
    if (numLayers <= 0) {
      throw std::runtime_error("Layer count must be positive.");
    }

    std::vector<Layer> loaded_layers;
    loaded_layers.reserve(static_cast<std::size_t>(numLayers));
    std::vector<std::vector<bool>> loaded_mask(
        static_cast<std::size_t>(numLayers));

    for (int expectedLayer = 0; expectedLayer < numLayers; ++expectedLayer) {
      if (!std::getline(file, line)) {
        throw std::runtime_error(
            "Unexpected EOF while reading layer metadata.");
      }
      std::vector<std::string> meta = splitCSVLine(trimCR(line));
      if (meta.size() != 4) {
        throw std::runtime_error("Invalid layer metadata line.");
      }

      const int layerIdx = parseStrictInt(meta[0]);
      const int numNeurons = parseStrictInt(meta[1]);
      const int numInputs = parseStrictInt(meta[2]);
      auto activation = createActivation(meta[3]);

      if (layerIdx != expectedLayer) {
        throw std::runtime_error("Layer metadata index mismatch.");
      }
      if (numNeurons <= 0 || numInputs <= 0) {
        throw std::runtime_error("Layer dimensions must be positive.");
      }
      if (!loaded_layers.empty()) {
        const int prevOutputs = static_cast<int>(loaded_layers.back().size());
        if (numInputs != prevOutputs) {
          throw std::runtime_error(
              "Layer input dimension mismatch in model metadata.");
        }
      }

      loaded_layers.emplace_back(numNeurons, numInputs, std::move(activation),
                                 rng);
      loaded_mask[static_cast<std::size_t>(layerIdx)].assign(
          static_cast<std::size_t>(numNeurons), false);
    }

    while (std::getline(file, line)) {
      line = trimCR(line);
      if (line.empty())
        continue;

      std::vector<std::string> cells = splitCSVLine(line);
      if (cells.size() < 4) {
        throw std::runtime_error("Malformed neuron line in model file.");
      }

      const int l = parseStrictInt(cells[0]);
      const int n = parseStrictInt(cells[1]);
      if (l < 0 || l >= numLayers) {
        throw std::out_of_range("Layer index out of bounds in model file.");
      }

      Layer &targetLayer = loaded_layers[static_cast<std::size_t>(l)];
      if (n < 0 || static_cast<std::size_t>(n) >= targetLayer.size()) {
        throw std::out_of_range("Neuron index out of bounds in model file.");
      }
      if (loaded_mask[static_cast<std::size_t>(l)]
                     [static_cast<std::size_t>(n)]) {
        throw std::runtime_error("Duplicate neuron entry in model file.");
      }

      const std::size_t expectedCells =
          3 + static_cast<std::size_t>(targetLayer.inputSize());
      if (cells.size() != expectedCells) {
        throw std::runtime_error("Weight count mismatch for neuron (" +
                                 std::to_string(l) + "," + std::to_string(n) +
                                 ").");
      }

      const double bias = parseStrictDouble(cells[2]);
      std::vector<double> weights(
          static_cast<std::size_t>(targetLayer.inputSize()));
      for (std::size_t w = 0; w < weights.size(); ++w) {
        weights[w] = parseStrictDouble(cells[3 + w]);
      }

      targetLayer.setNeuronParameters(static_cast<std::size_t>(n), bias,
                                      weights);
      loaded_mask[static_cast<std::size_t>(l)][static_cast<std::size_t>(n)] =
          true;
    }

    for (std::size_t l = 0; l < loaded_mask.size(); ++l) {
      for (std::size_t n = 0; n < loaded_mask[l].size(); ++n) {
        if (!loaded_mask[l][n]) {
          throw std::runtime_error("Incomplete model file: missing neuron (" +
                                   std::to_string(l) + "," + std::to_string(n) +
                                   ").");
        }
      }
    }

    layers = std::move(loaded_layers);
    std::cout << "Model loaded from " << filename << "\n";
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error loading model from " << filename << ": " << e.what()
              << "\n";
    return false;
  }
}

} // namespace miniann
