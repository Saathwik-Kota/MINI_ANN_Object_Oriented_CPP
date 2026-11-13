#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataLoader.hpp"
#include "MathVector.hpp"
#include "Matrix.hpp"
#include "Metrics.hpp"
#include "NeuralNetwork.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

namespace {

int g_testsRun = 0;
int g_testsFailed = 0;

void fail(const std::string& testName, const std::string& msg) {
    std::cerr << "[FAIL] " << testName << ": " << msg << "\n";
    ++g_testsFailed;
}

#define RUN_TEST(fn)                                                     \
    do {                                                                 \
        ++g_testsRun;                                                    \
        const int beforeFailures = g_testsFailed;                        \
        try {                                                            \
            fn();                                                        \
            if (g_testsFailed == beforeFailures) {                       \
                std::cout << "[PASS] " << #fn << "\n";                   \
            }                                                            \
        } catch (const std::exception& e) {                              \
            fail(#fn, std::string("Uncaught exception: ") + e.what());   \
        } catch (...) {                                                  \
            fail(#fn, "Uncaught unknown exception");                     \
        }                                                                \
    } while (0)

#define ASSERT_TRUE(cond)                                                \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fail(__func__, std::string("Expected true: ") + #cond);      \
            return;                                                      \
        }                                                                \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_NEAR(actual, expected, tol)                               \
    do {                                                                 \
        const double _actual_val = (actual);                             \
        const double _expected_val = (expected);                         \
        if (std::isnan(_actual_val) ||                                   \
            std::fabs(_actual_val - _expected_val) > (tol)) {            \
            std::ostringstream oss;                                      \
            oss << "Expected " << #actual << " (" << _actual_val         \
                << ") ~= " << _expected_val << " within " << (tol);      \
            fail(__func__, oss.str());                                   \
            return;                                                      \
        }                                                                \
    } while (0)

#define ASSERT_THROW(expr, ExcType)                                      \
    do {                                                                 \
        bool caught = false;                                             \
        try {                                                            \
            expr;                                                        \
        } catch (const ExcType&) {                                       \
            caught = true;                                               \
        } catch (...) {                                                  \
            fail(__func__, std::string("Wrong exception type for ") + #expr); \
            return;                                                      \
        }                                                                \
        if (!caught) {                                                   \
            fail(__func__, std::string("Expected exception ") + #ExcType + " for " + #expr); \
            return;                                                      \
        }                                                                \
    } while (0)

void testMathVectorOpsAndBounds() {
    using namespace miniann;

    MathVector a({1.0, 2.0, 3.0});
    MathVector b({4.0, 5.0, 6.0});

    ASSERT_NEAR(a * b, 32.0, 1e-12);

    MathVector sum = a + b;
    ASSERT_NEAR(sum.data[0], 5.0, 1e-12);
    ASSERT_NEAR(sum.data[1], 7.0, 1e-12);
    ASSERT_NEAR(sum.data[2], 9.0, 1e-12);

    MathVector diff = b - a;
    ASSERT_NEAR(diff.data[0], 3.0, 1e-12);
    ASSERT_NEAR(diff.data[1], 3.0, 1e-12);
    ASSERT_NEAR(diff.data[2], 3.0, 1e-12);

    MathVector scaled = a * 2.5;
    ASSERT_NEAR(scaled.data[1], 5.0, 1e-12);

    MathVector wrongSize({1.0, 2.0});
    ASSERT_THROW(static_cast<void>(a * wrongSize), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(a + wrongSize), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(a - wrongSize), std::invalid_argument);
    ASSERT_THROW(MathVector(-1, 0.0), std::invalid_argument);
}

void testActivationValuesAndStability() {
    using namespace miniann;

    Sigmoid sig;
    ASSERT_NEAR(sig.activate(0.0), 0.5, 1e-12);
    ASSERT_NEAR(sig.derivative(0.0), 0.25, 1e-12);
    // Extreme negative/positive inputs must remain finite and bounded
    const double sigNeg = sig.activate(-1000.0);
    const double sigPos = sig.activate(1000.0);
    ASSERT_TRUE(std::isfinite(sigNeg));
    ASSERT_TRUE(std::isfinite(sigPos));
    ASSERT_NEAR(sigNeg, 0.0, 1e-12);
    ASSERT_NEAR(sigPos, 1.0, 1e-12);

    ReLU relu;
    ASSERT_NEAR(relu.activate(-3.5), 0.0, 1e-12);
    ASSERT_NEAR(relu.activate(3.5), 3.5, 1e-12);
    ASSERT_NEAR(relu.derivative(-3.5), 0.0, 1e-12);
    ASSERT_NEAR(relu.derivative(3.5), 1.0, 1e-12);

    Tanh th;
    ASSERT_NEAR(th.activate(0.0), 0.0, 1e-12);
    ASSERT_NEAR(th.derivative(0.0), 1.0, 1e-12);

    ASSERT_THROW(createActivation("UnknownAct"), std::invalid_argument);
}

void testLossFunctionsAndStability() {
    using namespace miniann;

    MSE mse;
    ASSERT_NEAR(mse.calculate(0.8, 0.3), 0.125, 1e-12);
    ASSERT_NEAR(mse.derivative(0.8, 0.3), 0.5, 1e-12);

    BinaryCrossEntropy bce;
    // Extreme probabilities must be safely clamped without NaN/Inf
    ASSERT_TRUE(std::isfinite(bce.calculate(0.0, 1.0)));
    ASSERT_TRUE(std::isfinite(bce.calculate(1.0, 0.0)));
    ASSERT_TRUE(std::isfinite(bce.derivative(0.0, 1.0)));
    ASSERT_TRUE(std::isfinite(bce.derivative(1.0, 0.0)));
    ASSERT_NEAR(bce.calculate(0.9, 1.0), -std::log(0.9), 1e-12);
}

void verifyFiniteDifferenceGradients(miniann::NeuralNetwork& net,
                                     const miniann::MathVector& input,
                                     const miniann::MathVector& target,
                                     double tol) {
    net.forwardBackwardSample(input, target);

    constexpr double eps = 1e-6;
    for (std::size_t l = 0; l < net.layerCount(); ++l) {
        auto& layer = net.layerAt(l);
        for (std::size_t n = 0; n < layer.size(); ++n) {
            auto neuron = layer.neuronAt(n);
            const double origBias = neuron.getBias();
            const std::vector<double> origWeights = neuron.getWeights().data;
            const double analyticalBiasGrad = neuron.getDelta();
            const std::vector<double> analyticalWeightGrads =
                (neuron.getLastInputs() * neuron.getDelta()).data;

            // Check bias gradient
            neuron.setParameters(origBias + eps, origWeights);
            const double lossPlusB = net.computeLoss(net.predict(input), target);
            neuron.setParameters(origBias - eps, origWeights);
            const double lossMinusB = net.computeLoss(net.predict(input), target);
            neuron.setParameters(origBias, origWeights);

            const double numericBiasGrad = (lossPlusB - lossMinusB) / (2.0 * eps);
            ASSERT_NEAR(analyticalBiasGrad, numericBiasGrad, tol);

            // Check every weight gradient
            std::vector<double> perturbed = origWeights;
            for (std::size_t w = 0; w < origWeights.size(); ++w) {
                perturbed[w] = origWeights[w] + eps;
                neuron.setParameters(origBias, perturbed);
                const double lossPlusW = net.computeLoss(net.predict(input), target);

                perturbed[w] = origWeights[w] - eps;
                neuron.setParameters(origBias, perturbed);
                const double lossMinusW = net.computeLoss(net.predict(input), target);

                perturbed[w] = origWeights[w];
                neuron.setParameters(origBias, origWeights);

                const double numericWeightGrad = (lossPlusW - lossMinusW) / (2.0 * eps);
                ASSERT_NEAR(analyticalWeightGrads[w], numericWeightGrad, tol);
            }
        }
    }
}

void testFiniteDifferenceGradientCheckMSE() {
    using namespace miniann;

    NeuralNetwork net(
        std::make_unique<SGD>(0.05),
        std::make_unique<MSE>(),
        1234);
    net.addLayer(4, 3, std::make_shared<Tanh>());
    net.addLayer(3, 4, std::make_shared<ReLU>());
    net.addLayer(2, 3, std::make_shared<Sigmoid>());

    const MathVector input({0.45, -0.72, 0.31});
    const MathVector target({0.8, 0.2});
    verifyFiniteDifferenceGradients(net, input, target, 1e-6);
}

void testFiniteDifferenceGradientCheckBCE() {
    using namespace miniann;

    NeuralNetwork net(
        std::make_unique<Momentum>(0.01, 0.9),
        std::make_unique<BinaryCrossEntropy>(),
        5678);
    net.addLayer(4, 3, std::make_shared<Tanh>());
    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    const MathVector input(std::vector<double>{0.65, 0.15, -0.40});
    const MathVector target(std::vector<double>{1.0});
    verifyFiniteDifferenceGradients(net, input, target, 1e-6);
}

void testSerializationRoundTripAndValidation() {
    using namespace miniann;

    const std::string modelPath = "test_tmp_model.csv";
    const std::string badPath = "test_tmp_bad_model.csv";
    std::remove(modelPath.c_str());
    std::remove(badPath.c_str());

    NeuralNetwork net(
        std::make_unique<Momentum>(0.05, 0.9),
        std::make_unique<MSE>(),
        99);
    net.addLayer(4, 2, std::make_shared<Tanh>());
    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    const std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const std::vector<std::vector<double>> y = {{0.0}, {1.0}, {1.0}, {0.0}};
    net.train(X, y, 200);

    ASSERT_TRUE(net.saveModel(modelPath));

    NeuralNetwork loadedNet;
    ASSERT_TRUE(loadedNet.loadModel(modelPath));
    ASSERT_TRUE(loadedNet.layerCount() == net.layerCount());

    for (const auto& sample : X) {
        const auto expected = net.predict(sample);
        const auto actual = loadedNet.predict(sample);
        ASSERT_NEAR(actual[0], expected[0], 1e-12);
    }

    // Missing file fails cleanly
    ASSERT_FALSE(loadedNet.loadModel("non_existent_model_file.csv"));

    // Malformed header fails cleanly without corrupting loadedNet
    {
        std::ofstream bad(badPath);
        bad << "WRONG_HEADER\n1\n";
    }
    ASSERT_FALSE(loadedNet.loadModel(badPath));
    ASSERT_TRUE(loadedNet.layerCount() == net.layerCount());

    // Truncated model fails cleanly
    {
        std::ofstream bad(badPath);
        bad << "MINIANN_V1\n1\n0,2,2,Sigmoid\n0,0,0.1,0.2,0.3\n";
    }
    ASSERT_FALSE(loadedNet.loadModel(badPath));

    std::remove(modelPath.c_str());
    std::remove(badPath.c_str());
}

void testXorConvergenceAndConstPredict() {
    using namespace miniann;

    const std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    const std::vector<std::vector<double>> y = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    NeuralNetwork net(
        std::make_unique<Momentum>(0.1, 0.9),
        std::make_unique<MSE>(),
        42);
    net.addLayer(4, 2, std::make_shared<Tanh>());
    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    net.train(X, y, 2000);

    const NeuralNetwork& constRef = net;
    double totalLoss = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        const auto pred = constRef.predict(X[i]);
        const int predictedClass = (pred[0] >= 0.5) ? 1 : 0;
        const int expectedClass = static_cast<int>(y[i][0]);
        ASSERT_TRUE(predictedClass == expectedClass);
        totalLoss += constRef.computeLoss(MathVector(pred), MathVector(y[i]));
    }
    ASSERT_TRUE((totalLoss / X.size()) < 0.01);
}

void testDataLoaderValidation() {
    using namespace miniann;

    const std::string csvPath = "test_tmp_data.csv";
    std::remove(csvPath.c_str());

    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;

    // Missing file
    ASSERT_FALSE(DataLoader::loadCSV("missing_file_xyz.csv", X, y, 1));

    // Valid CSV with header and CRLF line endings
    {
        std::ofstream f(csvPath, std::ios::binary);
        f << "f1,f2,label\r\n0.1,0.2,1\r\n0.3,0.4,0\r\n";
    }
    ASSERT_TRUE(DataLoader::loadCSV(csvPath, X, y, 1, true));
    ASSERT_TRUE(X.size() == 2);
    ASSERT_TRUE(y.size() == 2);
    ASSERT_NEAR(X[0][0], 0.1, 1e-12);
    ASSERT_NEAR(y[1][0], 0.0, 1e-12);

    // Invalid numTargets
    ASSERT_FALSE(DataLoader::loadCSV(csvPath, X, y, 0, true));

    // Non-numeric cell rejected
    {
        std::ofstream f(csvPath);
        f << "0.1,abc,1\n";
    }
    ASSERT_FALSE(DataLoader::loadCSV(csvPath, X, y, 1, false));

    // Inconsistent column count rejected
    {
        std::ofstream f(csvPath);
        f << "0.1,0.2,1\n0.3,0.4\n";
    }
    ASSERT_FALSE(DataLoader::loadCSV(csvPath, X, y, 1, false));

    std::remove(csvPath.c_str());
}

void testLayerDimensionValidation() {
    using namespace miniann;

    NeuralNetwork net(
        std::make_unique<SGD>(0.1),
        std::make_unique<MSE>(),
        7);
    net.addLayer(3, 2, std::make_shared<Tanh>());
    ASSERT_THROW(net.addLayer(1, 5, std::make_shared<Sigmoid>()), std::invalid_argument);
    ASSERT_THROW(net.predict(std::vector<double>{1.0}), std::invalid_argument);
}

void testMatrixGemmAndOperations() {
    using namespace miniann;

    // A: 2x3, B: 3x2 -> C = A*B: 2x2
    Matrix A(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
    Matrix B(3, 2, std::vector<double>{7, 8, 9, 10, 11, 12});
    Matrix C = Matrix::multiply(A, B, false, false);
    ASSERT_TRUE(C.rows() == 2 && C.cols() == 2);
    ASSERT_NEAR(C(0, 0), 58.0, 1e-12);
    ASSERT_NEAR(C(0, 1), 64.0, 1e-12);
    ASSERT_NEAR(C(1, 0), 139.0, 1e-12);
    ASSERT_NEAR(C(1, 1), 154.0, 1e-12);

    // Transposed B: A(2x3) * B_T^T(3x2) where B_T is (2x3)
    Matrix B_T(2, 3, std::vector<double>{7, 9, 11, 8, 10, 12});
    Matrix C_transB = Matrix::multiply(A, B_T, false, true);
    ASSERT_NEAR(C_transB(0, 0), 58.0, 1e-12);
    ASSERT_NEAR(C_transB(1, 1), 154.0, 1e-12);

    // Transposed A: A^T(3x2) * A(2x3) -> 3x3
    Matrix C_transA = Matrix::multiply(A, A, true, false);
    ASSERT_TRUE(C_transA.rows() == 3 && C_transA.cols() == 3);
    ASSERT_NEAR(C_transA(0, 0), 17.0, 1e-12); // 1^2 + 4^2
    ASSERT_NEAR(C_transA(1, 2), 36.0, 1e-12); // 2*3 + 5*6

    // Dimension mismatch throws
    ASSERT_THROW(Matrix::multiply(A, A, false, false), std::invalid_argument);
}

void testMiniBatchTrainingConvergence() {
    using namespace miniann;

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

    NeuralNetwork net(
        std::make_unique<Momentum>(0.2, 0.9),
        std::make_unique<MSE>(),
        101);
    net.addLayer(6, 2, std::make_shared<Tanh>());
    net.addLayer(1, 6, std::make_shared<Sigmoid>());

    // Train with mini-batch size 2
    net.train(X, y, 1500, 2);

    const Matrix preds = net.predict(X);
    ASSERT_TRUE(preds.rows() == 4 && preds.cols() == 1);
    ASSERT_TRUE(preds(0, 0) < 0.15);
    ASSERT_TRUE(preds(1, 0) > 0.85);
    ASSERT_TRUE(preds(2, 0) > 0.85);
    ASSERT_TRUE(preds(3, 0) < 0.15);
}

void testTrainTestSplitAndOneHot() {
    using namespace miniann;

    Matrix X(100, 4, 1.0);
    Matrix y(100, 1, 0.0);
    for (std::size_t i = 0; i < 100; ++i) {
        y(i, 0) = static_cast<double>(i % 3);
    }

    DatasetSplit split = DataLoader::trainTestSplit(X, y, 0.2, 0.1, 42);
    ASSERT_TRUE(split.X_train.rows() == 70);
    ASSERT_TRUE(split.y_train.rows() == 70);
    ASSERT_TRUE(split.X_val.rows() == 10);
    ASSERT_TRUE(split.y_val.rows() == 10);
    ASSERT_TRUE(split.X_test.rows() == 20);
    ASSERT_TRUE(split.y_test.rows() == 20);

    // One-hot encoding
    Matrix labels(3, 1, std::vector<double>{0.0, 1.0, 2.0});
    Matrix encoded = DataLoader::oneHotEncode(labels, 3);
    ASSERT_TRUE(encoded.rows() == 3 && encoded.cols() == 3);
    ASSERT_NEAR(encoded(0, 0), 1.0, 1e-12);
    ASSERT_NEAR(encoded(0, 1), 0.0, 1e-12);
    ASSERT_NEAR(encoded(1, 1), 1.0, 1e-12);
    ASSERT_NEAR(encoded(2, 2), 1.0, 1e-12);

    // Out-of-range label throws
    Matrix badLabels(1, 1, std::vector<double>{5.0});
    ASSERT_THROW(DataLoader::oneHotEncode(badLabels, 3), std::out_of_range);
}

void testClassificationMetricsReport() {
    using namespace miniann;

    // Binary test: 4 samples
    // actual: [1, 1, 0, 0]
    // preds:  [0.9, 0.2, 0.1, 0.8] -> [1, 0, 0, 1]
    // TP=1, FN=1, TN=1, FP=1 -> Acc = 2/4 = 0.50
    Matrix preds(4, 1, std::vector<double>{0.9, 0.2, 0.1, 0.8});
    Matrix targets(4, 1, std::vector<double>{1.0, 1.0, 0.0, 0.0});

    ClassificationMetrics binMetrics = Metrics::evaluateBinary(preds, targets, 0.5);
    ASSERT_NEAR(binMetrics.accuracy, 0.50, 1e-12);
    ASSERT_NEAR(binMetrics.precision, 0.50, 1e-12);
    ASSERT_NEAR(binMetrics.recall, 0.50, 1e-12);
    ASSERT_NEAR(binMetrics.f1Score, 0.50, 1e-12);
    ASSERT_TRUE(binMetrics.confusionMatrix[1][1] == 1); // TP
    ASSERT_TRUE(binMetrics.confusionMatrix[1][0] == 1); // FN
    ASSERT_TRUE(binMetrics.confusionMatrix[0][0] == 1); // TN
    ASSERT_TRUE(binMetrics.confusionMatrix[0][1] == 1); // FP

    // Multi-class test: 3 samples, 3 classes
    Matrix mcPreds = Matrix::from2D({
        {0.8, 0.1, 0.1}, // class 0 (correct)
        {0.2, 0.7, 0.1}, // class 1 (correct)
        {0.3, 0.4, 0.3}  // class 1 predicted, but actual 2 (wrong)
    });
    Matrix mcTargets = Matrix::from2D({
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    });

    ClassificationMetrics mcMetrics = Metrics::evaluateMultiClass(mcPreds, mcTargets);
    ASSERT_NEAR(mcMetrics.accuracy, 2.0 / 3.0, 1e-12);
    ASSERT_TRUE(mcMetrics.confusionMatrix[0][0] == 1);
    ASSERT_TRUE(mcMetrics.confusionMatrix[1][1] == 1);
    ASSERT_TRUE(mcMetrics.confusionMatrix[2][1] == 1); // True 2 predicted as 1
}

void testSoftmaxAndCategoricalCrossEntropy() {
    using namespace miniann;

    Softmax sm;
    // Logits: extreme scale invariance test (should not NaN/overflow)
    Matrix Z = Matrix::from2D({
        {1000.0, 1002.0, 1000.0},
        {0.0, 0.0, 0.0}
    });
    Matrix A;
    sm.apply(Z, A);
    ASSERT_TRUE(A.rows() == 2 && A.cols() == 3);

    // Sum of probabilities across each row must equal 1.0
    ASSERT_NEAR(A(0, 0) + A(0, 1) + A(0, 2), 1.0, 1e-12);
    ASSERT_NEAR(A(1, 0) + A(1, 1) + A(1, 2), 1.0, 1e-12);
    ASSERT_TRUE(A(0, 1) > A(0, 0)); // logit 1002 > 1000

    CategoricalCrossEntropy cce;
    Matrix targets = Matrix::from2D({
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}
    });

    const double loss = cce.calculateBatch(A, targets);
    ASSERT_TRUE(std::isfinite(loss));
    ASSERT_TRUE(loss > 0.0);
}

void testAdamAndMultiClassConvergence() {
    using namespace miniann;

    // 3-Class dataset: 6 samples, 2 features
    const Matrix X = Matrix::from2D({
        {1.0, 1.0}, {1.2, 0.9}, // Class 0
        {-1.0, 1.0}, {-0.9, 1.1}, // Class 1
        {0.0, -1.0}, {0.1, -1.2}  // Class 2
    });
    const Matrix y = Matrix::from2D({
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}
    });

    NeuralNetwork net(
        std::make_unique<Adam>(0.05, 0.9, 0.999, 1e-8, 1e-4),
        std::make_unique<CategoricalCrossEntropy>(),
        42);

    net.addLayer(8, 2, std::make_shared<Tanh>());
    net.addLayer(3, 8, std::make_shared<Softmax>());

    TrainingConfig config;
    config.epochs = 300;
    config.batchSize = 2;
    config.verbose = false;

    net.train(X, y, Matrix(), Matrix(), config);

    Matrix preds = net.predict(X);
    ClassificationMetrics metrics = Metrics::evaluateMultiClass(preds, y);
    ASSERT_TRUE(metrics.accuracy == 1.0); // 100% convergence
}

} // namespace

int main() {
    RUN_TEST(testMathVectorOpsAndBounds);
    RUN_TEST(testActivationValuesAndStability);
    RUN_TEST(testLossFunctionsAndStability);
    RUN_TEST(testFiniteDifferenceGradientCheckMSE);
    RUN_TEST(testFiniteDifferenceGradientCheckBCE);
    RUN_TEST(testSerializationRoundTripAndValidation);
    RUN_TEST(testXorConvergenceAndConstPredict);
    RUN_TEST(testDataLoaderValidation);
    RUN_TEST(testLayerDimensionValidation);
    RUN_TEST(testMatrixGemmAndOperations);
    RUN_TEST(testMiniBatchTrainingConvergence);
    RUN_TEST(testTrainTestSplitAndOneHot);
    RUN_TEST(testClassificationMetricsReport);
    RUN_TEST(testSoftmaxAndCategoricalCrossEntropy);
    RUN_TEST(testAdamAndMultiClassConvergence);

    std::cout << "\nSummary: " << (g_testsRun - g_testsFailed)
              << "/" << g_testsRun << " tests passed.\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
