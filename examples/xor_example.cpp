#include <iomanip>
#include <iostream>
#include <memory>
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

int main() {
    using namespace miniann;

    std::cout << "=== MiniANN Example: XOR Convergence Demonstration ===\n";

    const Matrix X = Matrix::from2D({
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    });

    const Matrix y = Matrix::from2D({
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    });

    constexpr unsigned int kSeed = 42;
    NeuralNetwork net(
        std::make_unique<Momentum>(0.15, 0.9),
        std::make_unique<MSE>(),
        kSeed);

    net.addLayer(4, 2, std::make_shared<Tanh>());
    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    std::cout << "Training network on XOR logic gate...\n";
    net.train(X, y, 2000, 4);

    const Matrix preds = net.predict(X);
    std::cout << "\nPredictions:\n";
    for (std::size_t i = 0; i < 4; ++i) {
        const int predictedClass = (preds(i, 0) >= 0.5) ? 1 : 0;
        std::cout << "  Input: (" << X(i, 0) << ", " << X(i, 1)
                  << ") -> Expected: " << y(i, 0)
                  << " | Probability: " << std::fixed << std::setprecision(4) << preds(i, 0)
                  << " -> Class: " << predictedClass << "\n";
    }

    return 0;
}

