#include <iostream>
#include <memory>
#include "DataLoader.hpp"
#include "Metrics.hpp"
#include "NeuralNetwork.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

int main() {
    using namespace miniann;

    std::cout << "=== MiniANN Example: Custom Architecture & Early Stopping ===\n";

    // 1. Load CSV data into contiguous Matrix buffers
    Matrix X, y;
    if (!DataLoader::loadCSV("data/Data.csv", X, y, 1)) {
        return 1;
    }

    // 2. Train / Val / Test split
    DatasetSplit split = DataLoader::trainTestSplit(X, y, 0.20, 0.10, 42);

    // 3. Define network topology with Adam and Dropout
    NeuralNetwork net(
        std::make_unique<Adam>(0.01, 0.9, 0.999, 1e-8, 1e-4),
        std::make_unique<BinaryCrossEntropy>(),
        42);

    net.addLayer(16, static_cast<int>(split.X_train.cols()), std::make_shared<Tanh>(), 0.1); // 10% dropout
    net.addLayer(8, 16, std::make_shared<Tanh>());
    net.addLayer(1, 8, std::make_shared<Sigmoid>());

    // 4. Train with Early Stopping
    TrainingConfig config;
    config.epochs = 300;
    config.batchSize = 32;
    config.patience = 25;
    config.logInterval = 50;

    TrainingHistory history = net.train(split.X_train, split.y_train, split.X_val, split.y_val, config);
    std::cout << "Training stopped at epoch " << history.stoppedEpoch << "\n";

    // 5. Evaluate on held-out test data
    Matrix testPreds = net.predict(split.X_test);
    ClassificationMetrics metrics = Metrics::evaluate(testPreds, split.y_test, 0.5);
    Metrics::printReport(metrics, std::cout);

    return 0;
}

