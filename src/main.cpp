#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include "DataLoader.hpp"
#include "Metrics.hpp"
#include "NeuralNetwork.hpp"

int main() {
    using namespace miniann;

    const std::string dataset = "data/Data.csv";
    const int targetColumns = 1;

    std::cout << "========================================================\n";
    std::cout << "           MiniANN Production Training Pipeline         \n";
    std::cout << "========================================================\n\n";

    std::cout << "Loading dataset from " << dataset << "...\n";
    Matrix X, y;
    if (!DataLoader::loadCSV(dataset, X, y, targetColumns)) {
        return 1;
    }

    std::cout << "Successfully loaded " << X.rows() << " rows with "
              << X.cols() << " features and " << y.cols() << " target(s).\n\n";

    // 1. Train / Validation / Test Split (70% Train, 10% Validation, 20% Test)
    std::cout << "Splitting dataset (70% Train / 10% Val / 20% Test, seed=42)...\n";
    DatasetSplit split = DataLoader::trainTestSplit(X, y, 0.20, 0.10, 42);

    std::cout << "  Training set   : " << split.X_train.rows() << " samples\n";
    std::cout << "  Validation set : " << split.X_val.rows() << " samples\n";
    std::cout << "  Test set       : " << split.X_test.rows() << " samples\n\n";

    // 2. Build Architecture with Adam optimizer + L2 Weight Decay
    constexpr unsigned int kSeed = 42;
    constexpr double kLearningRate = 0.01;
    constexpr double kWeightDecay = 1e-4; // L2 Regularization

    NeuralNetwork myNet(
        std::make_unique<Adam>(kLearningRate, 0.9, 0.999, 1e-8, kWeightDecay),
        std::make_unique<BinaryCrossEntropy>(),
        kSeed);

    const int numFeatures = static_cast<int>(split.X_train.cols());
    const int numTargets = static_cast<int>(split.y_train.cols());

    myNet.addLayer(16, numFeatures, std::make_shared<Tanh>());
    myNet.addLayer(8, 16, std::make_shared<Tanh>());
    myNet.addLayer(numTargets, 8, std::make_shared<Sigmoid>());

    // 3. Configure Mini-Batch Training with Early Stopping & LR Scheduling
    TrainingConfig config;
    config.epochs = 500;
    config.batchSize = 32;
    config.patience = 40;        // Early stopping on validation loss
    config.lrDecayStep = 150;    // Step decay learning rate every 150 epochs
    config.lrDecayRate = 0.5;
    config.logInterval = 100;
    config.verbose = true;

    std::cout << "Starting Mini-Batch Training with Adam & Early Stopping...\n";
    TrainingHistory history = myNet.train(
        split.X_train, split.y_train,
        split.X_val, split.y_val,
        config);

    std::cout << "\nTraining finished at epoch " << history.stoppedEpoch
              << " (Early stopped: " << (history.earlyStopped ? "Yes" : "No") << ")\n\n";

    // 4. Rigorous Evaluation on Unseen Held-out Test Set
    std::cout << "Evaluating on Unseen Held-out Test Set (" << split.X_test.rows() << " samples)...\n";
    Matrix testPreds = myNet.predict(split.X_test);
    const double testLoss = myNet.computeLoss(testPreds, split.y_test) /
                            static_cast<double>(split.X_test.rows() * split.y_test.cols());

    std::cout << "Test Set Loss: " << std::fixed << std::setprecision(5) << testLoss << "\n";

    ClassificationMetrics testMetrics = Metrics::evaluate(testPreds, split.y_test, 0.5);
    Metrics::printReport(testMetrics, std::cout);

    // 5. Persist Model State
    const std::string modelPath = "data/trained_classification_model.csv";
    if (!myNet.saveModel(modelPath)) {
        std::cerr << "Failed to save trained model.\n";
        return 1;
    }

    return 0;
}
