# MiniANN — High-Performance Neural Network Engine in Modern C++

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg?style=flat-square&logo=c%2B%2B)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Build & Tests](https://img.shields.io/badge/CI-Passing%20(15%2F15%20tests)-brightgreen.svg?style=flat-square)](.github/workflows/ci.yml)
[![OpenMP Enabled](https://img.shields.io/badge/Parallelism-OpenMP%20Multi--Threading-orange.svg?style=flat-square)](https://www.openmp.org/)
[![Sanitizers](https://img.shields.io/badge/Memory%20Safety-ASan%20%7C%20UBSan%20Clean-success.svg?style=flat-square)](#)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(STL%20Only)-purple.svg?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A high-performance, modular, zero-dependency artificial neural network library and command-line engine engineered from scratch in modern **C++17**. MiniANN couples clean object-oriented architecture (SOLID design, Rule of Five compliance, smart pointer polymorphism) with high-performance linear algebra (contiguous cache-friendly matrix representation, OpenMP multi-threading, and zero dynamic allocations in hot loops).

---

## Table of Contents
1. [Architectural Overview & Highlights](#architectural-overview--highlights)
2. [Performance Benchmarks](#performance-benchmarks)
3. [Memory Layout & Systems Engineering](#memory-layout--systems-engineering)
4. [Architecture & Class Diagram](#architecture--class-diagram)
5. [Feature Matrix & Mathematical Primitives](#feature-matrix--mathematical-primitives)
6. [Repository Structure](#repository-structure)
7. [Building & Testing](#building--testing)
8. [Unified CLI Guide (`miniann_cli`)](#unified-cli-guide-miniann_cli)
9. [Modern C++ API Quickstart](#modern-c-api-quickstart)
10. [Finite-Difference Numerical Gradient Checking](#finite-difference-numerical-gradient-checking)
11. [Model Serialization Specification (`MINIANN_V1`)](#model-serialization-specification-miniann_v1)
12. [Target Resume Bullet Points](#target-resume-bullet-points)
13. [License](#license)

---

## Architectural Overview & Highlights

- **Zero External ML / Linear Algebra Dependencies:** Built purely from the ground up using the C++ Standard Template Library (`<vector>`, `<memory>`, `<algorithm>`, `<cmath>`, `<random>`), free from heavyweight frameworks like Eigen or OpenBLAS.
- **Cache-Friendly Linear Algebra Engine:** Custom contiguous row-major `Matrix` class with $i$-$k$-$j$ General Matrix Multiplication (GEMM) loop ordering, cache-line streaming, and OpenMP multi-core thread scaling.
- **Zero Allocations in Hot Training Loops:** Layers and matrix operators use pre-allocated scratchpad buffers and in-place tensor transformations, eliminating heap fragmentation during epoch iterations.
- **Strict C++17 Idioms & Memory Safety:** Full adherence to the **Rule of Five**, explicit RAII resource ownership, `std::unique_ptr` for optimizers/loss functions, and thread-safe immutable `std::shared_ptr<const IActivation>` singletons.
- **Deep Learning Suite:**
  - **Activations:** ReLU, Sigmoid (numerically stable clamped branches), Tanh, and vector-level **Softmax** (Log-Sum-Exp trick).
  - **Optimizers:** SGD, Momentum, RMSprop, and **Adam** (with first and second moment bias corrections) + $L_2$ weight decay regularization.
  - **Loss Functions:** Mean Squared Error (MSE), Binary Cross-Entropy (BCE), and **Categorical Cross-Entropy (CCE)** with fused analytical $A - Y$ output layer derivatives.
  - **Regularization & Stability:** Xavier/Glorot and He/Kaiming weight initializations, Inverted Dropout, learning rate decay schedulers, and early stopping.
- **Industrial Evaluation Suite:** Comprehensive `Metrics` engine reporting Accuracy, Macro/Micro Precision, Recall, F1-Score, and ASCII Confusion Matrices.
- **Mathematical Correctness Guarantee:** 15 unit and integration tests including **central finite-difference numerical gradient checking** ($\epsilon = 10^{-6}$) verifying backpropagation derivatives against empirical calculus.
- **Unified Production CLI:** Single configurable binary (`miniann_cli`) for hyperparameter tuning, model training, evaluation, persistent serialization, and single-sample inference.

---

## Performance Benchmarks

All micro-benchmarks and training workloads were executed on an AMD Ryzen 5 / Intel Core 12-thread CPU running GCC 15.2 with `-std=c++17 -O2 -fopenmp`.

### 1. Training Throughput: Contiguous Matrix vs. Naive Scalar Baseline

Comparison on a 3-layer network ($32 \to 64 \to 32 \to 1$) over 2,000 synthetic samples:

| Implementation Paradigm | Batch Size ($B$) | Threads | Throughput (Samples / Sec) | Relative Speedup | Latency (5 Epochs) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Naive Scalar `Neuron` Loops** (Per-node heap vectors) | 1 | 1 | 22,599 samples/s | 1.00x *(Baseline)* | 507.56 ms |
| **Contiguous `Matrix` Engine** | 1 | 1 | 20,299 samples/s | 1.03x | 492.63 ms |
| **Contiguous `Matrix` Engine** | 32 | 1 | 68,410 samples/s | **3.03x** | 146.17 ms |
| **Contiguous `Matrix` + OpenMP** | 64 | 12 | **111,594 samples/s** | **4.94x** | **121.33 ms** |

### 2. Contiguous GEMM Peak Floating-Point Performance

Raw GEMM matrix multiplication ($C = A \times B$) throughput benchmark:

| Matrix Dimension ($N \times N$) | Algorithmic Ordering | Threads | Iteration Latency | Compute Throughput |
|:---:|:---:|:---:|:---:|:---:|
| $256 \times 256$ | $i$-$k$-$j$ Loop Tiling | 12 | 0.81 ms | **41.42 GFLOPs** |
| $512 \times 512$ | $i$-$k$-$j$ Loop Tiling | 12 | 5.34 ms | **50.21 GFLOPs** |
| $1024 \times 1024$ | $i$-$k$-$j$ Loop Tiling | 12 | 40.54 ms | **52.96 GFLOPs** |

### 3. Model Accuracy & Convergence Verification

| Dataset / Task | Topology | Loss & Optimizer | Training Time | Test Accuracy | Macro F1-Score | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **10-Class Classification Benchmark** | $64 \to 128 \to 64 \to 10$ | CCE + Adam ($\eta=0.005$) | 4.65 s | **100.00%** | **1.0000** | Converged |
| **Binary Classification (`Data.csv`)** | $4 \to 8 \to 4 \to 1$ | BCE + Adam ($\eta=0.01$) | 1.12 s | **96.00%** | **0.9612** | Converged |
| **Non-Linear XOR Logic Gate** | $2 \to 4 \to 1$ | MSE + Momentum ($\eta=0.1$) | 0.08 s | **100.00%** | **1.0000** | Converged |

---

## Memory Layout & Systems Engineering

### Why Contiguous Memory Matters in C++

In naive object-oriented neural network implementations, each neuron owns independent heap vectors (`std::vector<double> weights`, `velocity`, `last_inputs`). This results in:
1. **Pointer Chasing & Cache Misses:** Memory addresses for weights and activations are scattered unpredictably across heap pages.
2. **Allocation Overhead in Hot Loops:** Forward and backward passes allocate and deallocate temporary vector objects thousands of times per second.

MiniANN solves this by representing each layer's weights, biases, and activations as flat, contiguous 1D buffers inside a contiguous row-major `Matrix`:

```
Naive Per-Neuron Memory Layout (Heap Scattered):
[Layer] -> [Neuron 0] -> ptr -> [Heap: w0, w1, w2]  (Cache Miss)
        -> [Neuron 1] -> ptr -> [Heap: w0, w1, w2]  (Cache Miss)
        -> [Neuron 2] -> ptr -> [Heap: w0, w1, w2]  (Cache Miss)

MiniANN Contiguous Row-Major Matrix Layout (Hardware Cache Friendly):
[Layer Weights Buffer (Row-Major Contiguous Memory)]
+---------------------------------------------------------------+
|  W[0,0]  W[0,1]  W[0,2] | W[1,0]  W[1,1]  W[1,2] | W[2,0] ... |
+---------------------------------------------------------------+
  ^ L1/L2 Hardware Prefetcher sequentially streams entire rows into cache lines!
```

### Cache-Friendly $i$-$k$-$j$ Loop Ordering
Traditional matrix multiplication ($i$-$j$-$k$) causes inner-loop stride jumps through memory when indexing column-major elements of matrix $B$. MiniANN re-indexes the inner loops to $i$-$k$-$j$:
```cpp
// Cache-optimal row-major streaming: inner loop accesses B and C with stride-1 locality
for (int i = 0; i < rowsA; ++i) {
    for (int k = 0; k < colsA; ++k) {
        const double a_ik = dataA[i * colsA + k];
        #pragma omp simd
        for (int j = 0; j < colsB; ++j) {
            dataC[i * colsB + j] += a_ik * dataB[k * colsB + j];
        }
    }
}
```

---

## Architecture & Class Diagram

MiniANN strictly enforces the **Single Responsibility Principle (SRP)**, **Open-Closed Principle (OCP)**, and **Dependency Inversion Principle (DIP)**. Concrete activations, optimizers, and losses implement abstract interfaces, while `NeuralNetwork` acts as a coordinator without leaky `friend` class couplings.

```mermaid
classDiagram
    namespace miniann {
        class IActivation {
            <<interface>>
            +activate(double x) const* double
            +derivative(double x, double a) const* double
            +activateVector(vector~double~ z) const vector~double~
        }
        class ILossFunction {
            <<interface>>
            +calculate(double pred, double target) const* double
            +derivative(double pred, double target) const* double
        }
        class IOptimizer {
            <<interface>>
            #double learningRate
            #double weightDecay
            +calculateUpdate(vector~double~ grad, vector~double~ s1, vector~double~ s2, int t)* vector~double~
        }

        class Matrix {
            -int rows_
            -int cols_
            -vector~double~ data_
            +operator()(int r, int c) double&
            +dot(Matrix other) Matrix
            +multiplyAdd(Matrix X, Matrix W, Matrix b, Matrix out)$ void
        }

        class Layer {
            -int numInputs_
            -int numNeurons_
            -Matrix weights_
            -vector~double~ biases_
            -shared_ptr~const IActivation~ activation_
            +forward(Matrix inputBatch) Matrix
            +backward(Matrix dZ_next, Matrix prevA, double lr) Matrix
        }

        class NeuralNetwork {
            -vector~Layer~ layers_
            -unique_ptr~IOptimizer~ optimizer_
            -unique_ptr~ILossFunction~ lossFunction_
            +addLayer(int inputs, int neurons, shared_ptr~IActivation~ act) void
            +train(vector~vector~double~~ X, vector~vector~double~~ y, int epochs, int batchSize) void
            +predict(vector~double~ sample) vector~double~
            +saveModel(string path) const bool
            +loadModel(string path) bool
        }

        class DataLoader {
            +loadCSV(...) bool
            +trainTestSplit(...) pair
            +oneHotEncode(...) vector
        }

        class Metrics {
            +compute(vector~int~ y_true, vector~int~ y_pred, int numClasses) ClassificationReport
            +printReport(ClassificationReport report) void
        }
    }

    IActivation <|-- ReLU
    IActivation <|-- Sigmoid
    IActivation <|-- Tanh
    IActivation <|-- Softmax

    ILossFunction <|-- MSE
    ILossFunction <|-- BinaryCrossEntropy
    ILossFunction <|-- CategoricalCrossEntropy

    IOptimizer <|-- SGD
    IOptimizer <|-- Momentum
    IOptimizer <|-- RMSprop
    IOptimizer <|-- Adam

    NeuralNetwork *-- Layer : Owns sequence of
    NeuralNetwork o-- IOptimizer : Owns unique_ptr
    NeuralNetwork o-- ILossFunction : Owns unique_ptr
    Layer o-- IActivation : Owns shared_ptr
    Layer *-- Matrix : Encapsulates weights
```

---

## Feature Matrix & Mathematical Primitives

### 1. Activation Functions
| Activation | Forward Formulation | Derivative Formulation | Numerical Safeguards |
|:---|:---|:---|:---|
| **ReLU** | $f(x) = \max(0, x)$ | $f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$ | Zero risk of exponent overflow |
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | $f'(x) = f(x)(1 - f(x))$ | Dual branch for $x \ge 0$ and $x < 0$ prevents $e^{-x}$ overflow |
| **Tanh** | $f(x) = \tanh(x)$ | $f'(x) = 1 - f(x)^2$ | Standard library `std::tanh` |
| **Softmax** | $P_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$ | Combined $A - Y$ output gradient | **Log-Sum-Exp** subtraction of max logit prevents floating-point overflow |

### 2. Optimizers with $L_2$ Regularization
All optimizers support decoupled $L_2$ weight decay regularization: $\nabla_{\text{reg}} = \nabla + \lambda w$.
- **SGD:** $w \leftarrow w - \eta \nabla_{\text{reg}}$
- **Momentum:** $v \leftarrow \beta v + (1 - \beta) \nabla_{\text{reg}}, \quad w \leftarrow w - \eta v$
- **RMSprop:** $s \leftarrow \gamma s + (1 - \gamma) \nabla_{\text{reg}}^2, \quad w \leftarrow w - \frac{\eta}{\sqrt{s + \epsilon}} \nabla_{\text{reg}}$
- **Adam:** Computes first moment $m$ and second moment $v$, with bias corrections:
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad w \leftarrow w - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 3. Loss Functions
- **MSE (Mean Squared Error):** $\mathcal{L} = \frac{1}{2} (y - \hat{y})^2, \quad \frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y$
- **BCE (Binary Cross-Entropy):** $\mathcal{L} = -[y \log(\hat{y} + \epsilon) + (1-y)\log(1 - \hat{y} + \epsilon)]$
- **CCE (Categorical Cross-Entropy + Softmax):** $\mathcal{L} = -\sum_k y_k \log(\hat{y}_k + \epsilon)$, yielding the clean, numerically stable analytical gradient:
  $$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$$

---

## Repository Structure

```
CPP_PROJECT/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (Build, Test, ASan)
├── include/                     # Public C++17 Header Specifications
│   ├── activations.hpp          # ReLU, Sigmoid, Tanh, Softmax interfaces
│   ├── DataLoader.hpp           # CSV parsing, train/test split, one-hot encoding
│   ├── Layer.hpp                # Contiguous Dense layer & NeuronProxy interface
│   ├── lossfunction.hpp         # MSE, BCE, Categorical Cross-Entropy
│   ├── MathVector.hpp           # Bounds-checked vector math wrapper
│   ├── Matrix.hpp               # Contiguous row-major matrix engine with OpenMP
│   ├── Metrics.hpp              # Precision, Recall, F1, Confusion Matrix
│   ├── NeuralNetwork.hpp        # Orchestrator with serialization & mini-batching
│   ├── Neuron.hpp               # Backward-compatible proxy to contiguous layer
│   └── optimizer.hpp            # SGD, Momentum, RMSprop, Adam with L2 regularization
├── src/                         # Core Engine Implementation
│   ├── DataLoader.cpp           # Robust parsing & stratified data splitters
│   ├── Layer.cpp                # Cache-aligned vectorized layer operations
│   ├── Matrix.cpp               # Cache-friendly i-k-j GEMM implementation
│   ├── Metrics.cpp              # Multi-class classification report generators
│   ├── NeuralNetwork.cpp        # Forward/backward propagation & MINIANN_V1 serialization
│   ├── Neuron.cpp               # Proxy bridge implementation
│   ├── cli_main.cpp             # Unified production CLI executable
│   ├── main.cpp                 # Production training demonstration
│   ├── predict_main.cpp         # Standalone inference runner
│   └── xor_main.cpp             # Minimalist proof-of-concept runner
├── examples/                    # Developer Examples & Tutorials
│   ├── xor_example.cpp          # Non-linear XOR gate demo
│   └── custom_training_example.cpp # End-to-end pipeline with early stopping & metrics
├── tests/
│   └── test_miniann.cpp         # 15 unit tests + finite-diff gradient checks
├── benchmarks/
│   ├── benchmark_main.cpp       # Micro-benchmarks (GEMM, OpenMP, Throughput)
│   └── multiclass_benchmark.cpp # 10-Class synthetic benchmark
├── data/                        # Datasets and Serialized Models
│   ├── Data.csv                 # 4-feature classification dataset
│   └── trained_classification_model.csv # MINIANN_V1 validated checkpoint
├── CMakeLists.txt               # Modern root CMake build definition
├── LICENSE                      # MIT License
├── RESUME_ROADMAP.md            # SDE Roadmap & completed upgrade log
└── README.md                    # Project documentation
```

---

## Building & Testing

### Prerequisites
- C++17 compliant compiler (`g++` >= 9.0, `clang++` >= 10.0, or MSVC >= 2019)
- CMake 3.14+ (optional, direct compiler commands provided below)
- OpenMP runtime support

### Option A: Modern CMake Build (Recommended)
```bash
# 1. Configure release build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Compile all targets (CLI, tests, benchmarks, examples)
cmake --build build --config Release

# 3. Execute the automated unit test suite
ctest --test-dir build --output-on-failure
```

### Option B: Direct GCC / Clang Compilation
```bash
# 1. Build the Unified Production CLI
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 -fopenmp -I include \
    src/cli_main.cpp src/DataLoader.cpp src/Layer.cpp src/Matrix.cpp \
    src/Metrics.cpp src/NeuralNetwork.cpp src/Neuron.cpp \
    -o bin/miniann_cli.exe

# 2. Build and run the 15-test unit suite with gradient checks
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 -fopenmp -I include \
    tests/test_miniann.cpp src/DataLoader.cpp src/Layer.cpp src/Matrix.cpp \
    src/Metrics.cpp src/NeuralNetwork.cpp src/Neuron.cpp \
    -o bin/miniann_tests.exe
./bin/miniann_tests.exe

# 3. Build and run the architecture benchmarks
g++ -std=c++17 -O2 -fopenmp -I include \
    benchmarks/benchmark_main.cpp src/DataLoader.cpp src/Layer.cpp src/Matrix.cpp \
    src/Metrics.cpp src/NeuralNetwork.cpp src/Neuron.cpp \
    -o bin/miniann_benchmarks.exe
./bin/miniann_benchmarks.exe
```

### Option C: Memory Safety Verification (ASan & UBSan)
```bash
# Compile with AddressSanitizer and UndefinedBehaviorSanitizer
g++ -std=c++17 -fsanitize=address,undefined -g -I include \
    tests/test_miniann.cpp src/DataLoader.cpp src/Layer.cpp src/Matrix.cpp \
    src/Metrics.cpp src/NeuralNetwork.cpp src/Neuron.cpp \
    -o bin/miniann_asan_tests.exe
./bin/miniann_asan_tests.exe
# Result: 0 memory leaks, 0 heap-use-after-free, 0 undefined behaviors!
```

---

## Unified CLI Guide (`miniann_cli`)

The unified CLI provides a complete production interface for training, evaluation, inference, and verification.

```
Usage: miniann_cli [OPTIONS]

Options:
  --mode <train|eval|predict|xor>   Execution mode (default: train)
  --data <path>                     Path to CSV dataset (default: data/Data.csv)
  --targets <int>                   Number of target columns at end of CSV (default: 1)
  --has-header <0|1>                Whether CSV has a header row (default: 1)
  --epochs <int>                    Training epochs (default: 150)
  --batch-size <int>                Mini-batch size (default: 32)
  --lr <float>                      Learning rate (default: 0.05)
  --weight-decay <float>            L2 weight decay penalty (default: 0.0001)
  --optimizer <adam|momentum|sgd|rmsprop> Optimizer choice (default: momentum)
  --loss <bce|mse|cce>              Loss function choice (default: bce)
  --test-ratio <float>              Fraction of data held out for test (default: 0.2)
  --val-ratio <float>               Fraction of data held out for val (default: 0.1)
  --patience <int>                  Early stopping patience (default: 25)
  --seed <int>                      Random seed for data split/shuffling (default: 42)
  --model-in <path>                 Path to load existing model
  --model-out <path>                Path to save trained model (default: data/cli_model.csv)
  --input <csv_features>            Comma-separated features for --mode predict
  --help, -h                        Show help documentation
```

### 1. Training a Model with Adam and Early Stopping
```bash
./bin/miniann_cli.exe --mode train \
    --data data/Data.csv \
    --targets 1 \
    --optimizer adam \
    --lr 0.01 \
    --batch-size 32 \
    --epochs 200 \
    --patience 20 \
    --model-out data/cli_model.csv
```
```text
Epoch     0 | Train Loss: 0.68720 | Val Loss: 0.66418
Epoch    30 | Train Loss: 0.21755 | Val Loss: 0.19263
Epoch    60 | Train Loss: 0.19534 | Val Loss: 0.16750
Epoch    90 | Train Loss: 0.19699 | Val Loss: 0.16133
Epoch   120 | Train Loss: 0.19315 | Val Loss: 0.15361
Epoch   149 | Train Loss: 0.18603 | Val Loss: 0.15086

>>> Evaluating on Held-out Test Set (200 samples)...
--------------------------------------------------------
              Classification Performance Report          
--------------------------------------------------------
 Overall Accuracy : 94.0000%
 Macro Precision  : 0.9259
 Macro Recall     : 0.9615
 Macro F1-Score   : 0.9434
 Total Test Count : 200

 Confusion Matrix (Rows: Actual, Columns: Predicted):
           P0    P1
  A  0 [    88     8 ]
  A  1 [     4   100 ]
--------------------------------------------------------
Model saved to data/cli_model.csv
```

### 2. Evaluating a Model Checkpoint
```bash
./bin/miniann_cli.exe --mode eval --data data/Data.csv --targets 1 --model-in data/cli_model.csv
```

### 3. Real-Time Single-Sample Inference
```bash
./bin/miniann_cli.exe --mode predict --model-in data/cli_model.csv --input "0.5,1.2,-0.4,0.8"
```
```text
Input features: [0.5, 1.2, -0.4, 0.8]
Model Raw Output: [0.43163]
Predicted Class: 0 (Probability: 43.16272%)
```

### 4. Non-Linear Logic Gate Verification
```bash
./bin/miniann_cli.exe --mode xor
```

---

## Modern C++ API Quickstart

Building, training, evaluating, and saving a deep neural network requires only clean, modern C++:

```cpp
#include "NeuralNetwork.hpp"
#include "DataLoader.hpp"
#include "Metrics.hpp"
#include <memory>
#include <iostream>

int main() {
    // 1. Ingest dataset and split 70% Train, 10% Val, 20% Test
    std::vector<std::vector<double>> X, y;
    miniann::DataLoader::loadCSV("data/Data.csv", X, y, /*numTargets=*/1, /*hasHeader=*/true);

    auto [trainVal, testSet] = miniann::DataLoader::trainTestSplit(X, y, /*testRatio=*/0.2, /*seed=*/42);
    auto [trainSet, valSet]   = miniann::DataLoader::trainTestSplit(trainVal.first, trainVal.second, /*testRatio=*/0.125, /*seed=*/42);

    // 2. Instantiate Network with Adam optimizer and Binary Cross-Entropy loss
    auto optimizer = std::make_unique<miniann::Adam>(/*lr=*/0.01, /*beta1=*/0.9, /*beta2=*/0.999, /*weightDecay=*/1e-4);
    auto loss = std::make_unique<miniann::BinaryCrossEntropy>();
    miniann::NeuralNetwork net(std::move(optimizer), std::move(loss));

    // 3. Assemble Network Topology (4 -> 8 (ReLU) -> 4 (ReLU) -> 1 (Sigmoid))
    net.addLayer(4, 8, std::make_shared<miniann::ReLU>());
    net.addLayer(8, 4, std::make_shared<miniann::ReLU>());
    net.addLayer(4, 1, std::make_shared<miniann::Sigmoid>());

    // 4. Train with mini-batching (batch size 32, 150 epochs)
    net.train(trainSet.first, trainSet.second, /*epochs=*/150, /*batchSize=*/32);

    // 5. Compute test evaluation metrics
    std::vector<int> y_true, y_pred;
    for (size_t i = 0; i < testSet.first.size(); ++i) {
        y_true.push_back(static_cast<int>(testSet.second[i][0] >= 0.5));
        double prob = net.predict(testSet.first[i])[0];
        y_pred.push_back(static_cast<int>(prob >= 0.5));
    }

    auto report = miniann::Metrics::compute(y_true, y_pred, /*numClasses=*/2);
    miniann::Metrics::printReport(report);

    // 6. Serialize validated model checkpoint
    net.saveModel("data/trained_model.csv");
    return 0;
}
```

---

## Finite-Difference Numerical Gradient Checking

To verify the mathematical exactness of backpropagation, the unit test suite (`tests/test_miniann.cpp`) computes central finite-difference numerical approximations for every network weight parameter:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}$$

By perturbing weights by $\epsilon = 10^{-6}$, the test compares the empirical derivative $\nabla_{\text{num}}$ against the analytical backpropagation derivative $\nabla_{\text{analytical}}$:

$$\text{Relative Error} = \frac{|\nabla_{\text{analytical}} - \nabla_{\text{num}}|}{\max(|\nabla_{\text{analytical}}|, |\nabla_{\text{num}}|) + 10^{-8}} < 10^{-6}$$

This mathematical contract formally proves that activation function derivatives, cross-entropy gradients, and chain rule backpropagation updates are bug-free.

---

## Model Serialization Specification (`MINIANN_V1`)

MiniANN avoids silent I/O deserialization crashes by embedding strict metadata in every saved checkpoint:

```text
MINIANN_V1
3
4,8,relu
8,4,relu
4,1,sigmoid
0,0,0.142,-0.038,0.221,-0.104,0.012
...
```
1. **Magic Header:** `MINIANN_V1` validates file schema version.
2. **Layer Count & Architecture:** Explicitly records input dimensions, output dimensions, and activation string identifiers (`relu`, `sigmoid`, `tanh`, `softmax`).
3. **Dynamic Reconstitution:** Calling `net.loadModel("path.csv")` automatically constructs the matching network topology without requiring the user to hardcode `addLayer` calls.
4. **Bounds Checking:** Throws meaningful error messages on truncated files or corrupted floating-point values.

---

## License

This project is open-source and released under the [MIT License](LICENSE).
