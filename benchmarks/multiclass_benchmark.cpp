#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "DataLoader.hpp"
#include "Metrics.hpp"
#include "NeuralNetwork.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

namespace miniann {

// Generates a reproducible 10-class handwritten-digit style dataset
// (5,000 samples, 64 features, 10 classes: 0 through 9)
void generate10ClassDataset(std::size_t numSamples,
                            std::size_t numFeatures,
                            std::size_t numClasses,
                            Matrix& X,
                            Matrix& yOneHot,
                            unsigned int seed = 42) {
    X.resize(numSamples, numFeatures);
    yOneHot.resize(numSamples, numClasses, 0.0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noiseDist(-0.15, 0.15);

    // Create 10 distinct class prototypes in 64-dimensional feature space
    std::vector<std::vector<double>> prototypes(numClasses, std::vector<double>(numFeatures, 0.0));
    for (std::size_t c = 0; c < numClasses; ++c) {
        for (std::size_t f = 0; f < numFeatures; ++f) {
            prototypes[c][f] = std::sin(static_cast<double>(c * 17 + f * 11) * 0.25);
        }
    }

    const std::size_t samplesPerClass = numSamples / numClasses;
    std::size_t rowIdx = 0;

    for (std::size_t c = 0; c < numClasses; ++c) {
        for (std::size_t s = 0; s < samplesPerClass; ++s) {
            for (std::size_t f = 0; f < numFeatures; ++f) {
                X(rowIdx, f) = prototypes[c][f] + noiseDist(rng);
            }
            yOneHot(rowIdx, c) = 1.0;
            ++rowIdx;
        }
    }
}

} // namespace miniann

int main() {
    using namespace miniann;

    std::cout << "=================================================================\n";
    std::cout << "   MiniANN 10-Class Multi-Layer Softmax + CCE + Adam Benchmark   \n";
    std::cout << "=================================================================\n\n";

    constexpr std::size_t kSamples = 5000;
    constexpr std::size_t kFeatures = 64;
    constexpr std::size_t kClasses = 10;
    constexpr unsigned int kSeed = 101;

    std::cout << "[Workload] Generating 10-Class Dataset (" << kSamples << " samples, "
              << kFeatures << " features, " << kClasses << " classes)...\n";
    Matrix X, y;
    generate10ClassDataset(kSamples, kFeatures, kClasses, X, y, kSeed);

    std::cout << "Splitting into 80% Train (4,000 samples) and 20% Held-out Test (1,000 samples)...\n";
    DatasetSplit split = DataLoader::trainTestSplit(X, y, 0.20, 0.0, kSeed);

    // Build Deep Multi-Class Network: 64 -> 128 (ReLU) -> 64 (Tanh) -> 10 (Softmax)
    std::cout << "Network Architecture: 64 -> 128 (ReLU) -> 64 (Tanh) -> 10 (Softmax)\n";
    std::cout << "Optimizer: Adam (lr=0.005, beta1=0.9, beta2=0.999, L2 weight_decay=1e-4)\n";
    std::cout << "Loss: CategoricalCrossEntropy\n\n";

    NeuralNetwork net(
        std::make_unique<Adam>(0.005, 0.9, 0.999, 1e-8, 1e-4),
        std::make_unique<CategoricalCrossEntropy>(),
        kSeed);

    net.addLayer(128, static_cast<int>(kFeatures), std::make_shared<ReLU>());
    net.addLayer(64, 128, std::make_shared<Tanh>());
    net.addLayer(static_cast<int>(kClasses), 64, std::make_shared<Softmax>());

    TrainingConfig config;
    config.epochs = 50;
    config.batchSize = 64;
    config.logInterval = 10;
    config.verbose = true;

    const auto tStart = std::chrono::high_resolution_clock::now();
    net.train(split.X_train, split.y_train, Matrix(), Matrix(), config);
    const auto tEnd = std::chrono::high_resolution_clock::now();
    const double trainDurationSec = std::chrono::duration<double>(tEnd - tStart).count();

    std::cout << "\nTraining Completed in " << std::fixed << std::setprecision(2)
              << trainDurationSec << " seconds.\n\n";

    // Evaluate on 1,000 Unseen Held-out Test Samples
    std::cout << ">>> Evaluating on Unseen Held-out Test Set (1,000 samples)...\n";
    Matrix testPreds = net.predict(split.X_test);

    ClassificationMetrics testMetrics = Metrics::evaluateMultiClass(testPreds, split.y_test);
    Metrics::printReport(testMetrics, std::cout);

    std::cout << "\n[Resume Result] Tested on 1,000 test samples: Accuracy = "
              << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0)
              << "% | Macro F1 = " << std::setprecision(4) << testMetrics.f1Score
              << " | Trained in " << std::setprecision(2) << trainDurationSec << "s.\n";

    return 0;
}

