#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "MathVector.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

namespace miniann {

// ======================= Naive Baseline Implementation =======================
// Models the unoptimized pre-P2 architecture with per-neuron objects and
// heap-allocated temporary MathVector instances in hot loops.

struct NaiveNeuron {
    MathVector weights;
    double bias{0.0};
    MathVector velocity;
    double bias_velocity{0.0};
    std::shared_ptr<const IActivation> activation;
    MathVector last_inputs;
    double last_z{0.0};
    double last_a{0.0};
    double delta{0.0};

    NaiveNeuron(int numInputs, int layerSize, std::shared_ptr<const IActivation> act, std::mt19937& rng)
        : weights(numInputs, 0.0), velocity(numInputs, 0.0), activation(act), last_inputs(numInputs, 0.0) {
        const double stddev = act->weightInitStdDev(numInputs, layerSize);
        std::normal_distribution<double> dis(0.0, stddev);
        for (int i = 0; i < numInputs; ++i) {
            weights.data[i] = dis(rng);
        }
    }

    double forward(const MathVector& inputs) {
        last_inputs = inputs;
        last_z = (weights * inputs) + bias;
        last_a = activation->activate(last_z);
        return last_a;
    }
};

struct NaiveLayer {
    std::vector<NaiveNeuron> neurons;
    NaiveLayer(int numNeurons, int numInputs, std::shared_ptr<const IActivation> act, std::mt19937& rng) {
        neurons.reserve(numNeurons);
        for (int i = 0; i < numNeurons; ++i) {
            neurons.emplace_back(numInputs, numNeurons, act, rng);
        }
    }

    MathVector forward(const MathVector& inputs) {
        std::vector<double> out(neurons.size());
        for (std::size_t i = 0; i < neurons.size(); ++i) {
            out[i] = neurons[i].forward(inputs);
        }
        return MathVector(out);
    }
};

class NaiveNetwork {
public:
    std::vector<NaiveLayer> layers;
    std::unique_ptr<IOptimizer> optimizer;
    std::unique_ptr<ILossFunction> loss;
    std::mt19937 rng;

    NaiveNetwork(std::unique_ptr<IOptimizer> opt, std::unique_ptr<ILossFunction> l, unsigned int seed)
        : optimizer(std::move(opt)), loss(std::move(l)), rng(seed) {}

    void addLayer(int numNeurons, int numInputs, std::shared_ptr<const IActivation> act) {
        layers.emplace_back(numNeurons, numInputs, act, rng);
    }

    MathVector predict(const MathVector& input) {
        MathVector cur = input;
        for (auto& layer : layers) {
            cur = layer.forward(cur);
        }
        return cur;
    }

    void trainEpoch(const std::vector<MathVector>& X, const std::vector<MathVector>& y) {
        for (std::size_t ex = 0; ex < X.size(); ++ex) {
            MathVector pred = predict(X[ex]);

            NaiveLayer& outLayer = layers.back();
            for (std::size_t i = 0; i < outLayer.neurons.size(); ++i) {
                NaiveNeuron& n = outLayer.neurons[i];
                double dL_da = loss->derivative(n.last_a, y[ex].data[i]);
                double da_dz = n.activation->derivative(n.last_z);
                n.delta = dL_da * da_dz;
            }

            for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
                NaiveLayer& hidden = layers[l];
                NaiveLayer& next = layers[l + 1];
                for (std::size_t i = 0; i < hidden.neurons.size(); ++i) {
                    double err = 0.0;
                    for (std::size_t j = 0; j < next.neurons.size(); ++j) {
                        err += next.neurons[j].delta * next.neurons[j].weights.data[i];
                    }
                    hidden.neurons[i].delta = err * hidden.neurons[i].activation->derivative(hidden.neurons[i].last_z);
                }
            }

            for (auto& layer : layers) {
                for (auto& n : layer.neurons) {
                    MathVector grads = n.last_inputs * n.delta;
                    MathVector wUp = optimizer->calculateUpdate(grads, n.velocity);
                    double bUp = optimizer->calculateUpdate(n.delta, n.bias_velocity);
                    n.weights = n.weights - wUp;
                    n.bias -= bUp;
                }
            }
        }
    }
};

} // namespace miniann

int main() {
    using namespace miniann;

    std::cout << "=================================================================\n";
    std::cout << "          MiniANN High-Performance Architecture Benchmark        \n";
    std::cout << "=================================================================\n\n";

#ifdef _OPENMP
    const int maxThreads = omp_get_max_threads();
    std::cout << "[System Info] OpenMP 201511 detected. Available threads: " << maxThreads << "\n";
#else
    const int maxThreads = 1;
    std::cout << "[System Info] OpenMP not active in this build.\n";
#endif

    // Generate realistic synthetic dataset (2000 samples, 32 input features)
    const std::size_t numSamples = 2000;
    const std::size_t inDim = 32;
    const std::size_t outDim = 1;
    const int benchmarkEpochs = 5;

    std::mt19937 dataGen(1337);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<MathVector> X_vec;
    std::vector<MathVector> y_vec;
    X_vec.reserve(numSamples);
    y_vec.reserve(numSamples);

    Matrix X_mat(numSamples, inDim);
    Matrix y_mat(numSamples, outDim);

    for (std::size_t i = 0; i < numSamples; ++i) {
        std::vector<double> rowX(inDim);
        for (std::size_t j = 0; j < inDim; ++j) rowX[j] = dist(dataGen);
        std::vector<double> rowY = {(rowX[0] * rowX[1] + rowX[2] > 0.0) ? 1.0 : 0.0};

        X_vec.emplace_back(rowX);
        y_vec.emplace_back(rowY);
        for (std::size_t j = 0; j < inDim; ++j) {
            X_mat(i, j) = rowX[j];
        }
        y_mat(i, 0) = rowY[0];
    }

    std::cout << "[Workload] Samples: " << numSamples << " | Features: " << inDim
              << " | Topology: 32 -> 64 (Tanh) -> 32 (Tanh) -> 1 (Sigmoid)\n";
    std::cout << "[Workload] Epochs: " << benchmarkEpochs << "\n\n";

    // -------------------------------------------------------------------------
    // Benchmark 1: Naive Scalar Architecture
    // -------------------------------------------------------------------------
    std::cout << ">>> Running Benchmark 1: Naive Scalar Architecture (Per-Neuron Objects)...\n";
    NaiveNetwork naiveNet(std::make_unique<Momentum>(0.01, 0.9), std::make_unique<MSE>(), 42);
    naiveNet.addLayer(64, static_cast<int>(inDim), std::make_shared<Tanh>());
    naiveNet.addLayer(32, 64, std::make_shared<Tanh>());
    naiveNet.addLayer(1, 32, std::make_shared<Sigmoid>());

    const auto tNaiveStart = std::chrono::high_resolution_clock::now();
    for (int ep = 0; ep < benchmarkEpochs; ++ep) {
        naiveNet.trainEpoch(X_vec, y_vec);
    }
    const auto tNaiveEnd = std::chrono::high_resolution_clock::now();
    const double naiveDurationMs =
        std::chrono::duration<double, std::milli>(tNaiveEnd - tNaiveStart).count();
    const double naiveThroughput =
        (static_cast<double>(numSamples * benchmarkEpochs) / (naiveDurationMs / 1000.0));

    std::cout << "    [Naive Scalar] Total Time: " << std::fixed << std::setprecision(2)
              << naiveDurationMs << " ms | Throughput: "
              << static_cast<long long>(naiveThroughput) << " samples/sec\n\n";

    // -------------------------------------------------------------------------
    // Benchmark 2: Contiguous Matrix Engine (Single-Threaded, Online SGD B=1)
    // -------------------------------------------------------------------------
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    std::cout << ">>> Running Benchmark 2: Contiguous Matrix Engine (Single-Threaded, Batch Size = 1)...\n";
    NeuralNetwork matrixNetSingle(std::make_unique<Momentum>(0.01, 0.9), std::make_unique<MSE>(), 42);
    matrixNetSingle.addLayer(64, static_cast<int>(inDim), std::make_shared<Tanh>());
    matrixNetSingle.addLayer(32, 64, std::make_shared<Tanh>());
    matrixNetSingle.addLayer(1, 32, std::make_shared<Sigmoid>());

    const auto tMatSingleStart = std::chrono::high_resolution_clock::now();
    matrixNetSingle.train(X_mat, y_mat, benchmarkEpochs, 1);
    const auto tMatSingleEnd = std::chrono::high_resolution_clock::now();
    const double matSingleDurationMs =
        std::chrono::duration<double, std::milli>(tMatSingleEnd - tMatSingleStart).count();
    const double matSingleThroughput =
        (static_cast<double>(numSamples * benchmarkEpochs) / (matSingleDurationMs / 1000.0));
    const double speedupOverNaive = naiveDurationMs / matSingleDurationMs;

    std::cout << "    [Matrix 1-Thread B=1] Total Time: " << matSingleDurationMs
              << " ms | Throughput: " << static_cast<long long>(matSingleThroughput)
              << " samples/sec | Speedup vs Naive: " << std::setprecision(2)
              << speedupOverNaive << "x\n\n";

    // -------------------------------------------------------------------------
    // Benchmark 3: Contiguous Matrix Engine (Mini-Batch B=64, Multi-Threaded OpenMP)
    // -------------------------------------------------------------------------
#ifdef _OPENMP
    omp_set_num_threads(maxThreads);
#endif
    std::cout << ">>> Running Benchmark 3: Contiguous Matrix Engine (Mini-Batch B=64, "
              << maxThreads << " OpenMP Threads)...\n";
    NeuralNetwork matrixNetParallel(std::make_unique<Momentum>(0.01, 0.9), std::make_unique<MSE>(), 42);
    matrixNetParallel.addLayer(64, static_cast<int>(inDim), std::make_shared<Tanh>());
    matrixNetParallel.addLayer(32, 64, std::make_shared<Tanh>());
    matrixNetParallel.addLayer(1, 32, std::make_shared<Sigmoid>());

    const auto tMatParStart = std::chrono::high_resolution_clock::now();
    matrixNetParallel.train(X_mat, y_mat, benchmarkEpochs, 64);
    const auto tMatParEnd = std::chrono::high_resolution_clock::now();
    const double matParDurationMs =
        std::chrono::duration<double, std::milli>(tMatParEnd - tMatParStart).count();
    const double matParThroughput =
        (static_cast<double>(numSamples * benchmarkEpochs) / (matParDurationMs / 1000.0));
    const double speedupParallelOverNaive = naiveDurationMs / matParDurationMs;

    std::cout << "    [Matrix OpenMP B=64] Total Time: " << matParDurationMs
              << " ms | Throughput: " << static_cast<long long>(matParThroughput)
              << " samples/sec | Total Speedup vs Naive: " << std::setprecision(2)
              << speedupParallelOverNaive << "x\n\n";

    // -------------------------------------------------------------------------
    // Benchmark 4: Raw GEMM Throughput (GFLOPs)
    // -------------------------------------------------------------------------
    std::cout << ">>> Running Benchmark 4: Contiguous GEMM Raw Throughput...\n";
    const std::vector<std::size_t> testSizes = {256, 512};
    for (std::size_t N : testSizes) {
        Matrix A(N, N, 1.01);
        Matrix B(N, N, 0.99);
        Matrix C(N, N, 0.0);

        const auto tGemmStart = std::chrono::high_resolution_clock::now();
        const int iters = 10;
        for (int it = 0; it < iters; ++it) {
            Matrix::gemm(A, B, C, false, false, 1.0, 0.0);
        }
        const auto tGemmEnd = std::chrono::high_resolution_clock::now();
        const double gemmDurationMs =
            std::chrono::duration<double, std::milli>(tGemmEnd - tGemmStart).count();

        // 2 * N^3 floating point operations per GEMM
        const double totalFlops = static_cast<double>(iters) * 2.0 * static_cast<double>(N * N * N);
        const double gflops = (totalFlops / (gemmDurationMs / 1000.0)) / 1e9;

        std::cout << "    GEMM (" << N << "x" << N << " x " << N << "x" << N << "): "
                  << std::setprecision(2) << (gemmDurationMs / iters)
                  << " ms/iter | Compute: " << gflops << " GFLOPs\n";
    }

    std::cout << "\n=================================================================\n";
    std::cout << "                    Benchmark Summary Table                      \n";
    std::cout << "=================================================================\n";
    std::cout << " Implementation                 | Time (ms)  | Speedup vs Baseline\n";
    std::cout << "--------------------------------+------------+--------------------\n";
    std::cout << " Naive Scalar Baseline (P0/P1)  | " << std::setw(10) << naiveDurationMs << " | 1.00x (Baseline)\n";
    std::cout << " Matrix Contiguous (1-Thread)   | " << std::setw(10) << matSingleDurationMs << " | " << speedupOverNaive << "x\n";
    std::cout << " Matrix Mini-Batch (OpenMP)     | " << std::setw(10) << matParDurationMs << " | " << speedupParallelOverNaive << "x\n";
    std::cout << "=================================================================\n";

    return 0;
}
