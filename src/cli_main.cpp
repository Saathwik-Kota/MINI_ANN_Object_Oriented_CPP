#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "DataLoader.hpp"
#include "Metrics.hpp"
#include "NeuralNetwork.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimizer.hpp"

namespace miniann {

struct CliArgs {
    std::string mode{"train"}; // train, eval, predict, xor
    std::string dataPath{"data/Data.csv"};
    int targetColumns{1};
    bool hasHeader{false};

    int epochs{300};
    std::size_t batchSize{32};
    double lr{0.01};
    double weightDecay{1e-4};
    std::string optimizerName{"adam"}; // adam, momentum, sgd, rmsprop
    std::string lossName{"bce"};       // bce, mse, cce

    double testRatio{0.20};
    double valRatio{0.10};
    int patience{30};
    unsigned int seed{42};

    std::string modelIn{"data/trained_classification_model.csv"};
    std::string modelOut{"data/trained_classification_model.csv"};

    std::vector<double> predictInput{0.8, 0.7, 0.6, 0.5};
};

void printUsage(const char* progName) {
    std::cout << "MiniANN CLI — High-Performance Neural Network Engine in Modern C++\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << progName << " [options]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  --mode <train|eval|predict|xor>       Execution mode (default: train)\n\n";
    std::cout << "Data Options:\n";
    std::cout << "  --data <path>                         Path to CSV dataset (default: data/Data.csv)\n";
    std::cout << "  --targets <int>                       Number of target columns at end (default: 1)\n";
    std::cout << "  --has-header                          CSV contains header row\n";
    std::cout << "  --test-ratio <float>                  Held-out test set ratio [0, 1) (default: 0.20)\n";
    std::cout << "  --val-ratio <float>                   Validation set ratio [0, 1) (default: 0.10)\n";
    std::cout << "  --seed <int>                          RNG seed for reproducibility (default: 42)\n\n";
    std::cout << "Training Hyperparameters:\n";
    std::cout << "  --epochs <int>                        Maximum training epochs (default: 300)\n";
    std::cout << "  --batch-size <int>                    Mini-batch size (default: 32)\n";
    std::cout << "  --lr <float>                          Initial learning rate (default: 0.01)\n";
    std::cout << "  --weight-decay <float>                L2 regularization factor (default: 0.0001)\n";
    std::cout << "  --optimizer <adam|momentum|sgd|rmsprop> (default: adam)\n";
    std::cout << "  --loss <bce|mse|cce>                  Loss function (default: bce)\n";
    std::cout << "  --patience <int>                      Early stopping patience (default: 30, 0=off)\n\n";
    std::cout << "Model I/O & Inference:\n";
    std::cout << "  --model-in <path>                     Trained model file to load (default: data/trained_classification_model.csv)\n";
    std::cout << "  --model-out <path>                    Output model path (default: data/trained_classification_model.csv)\n";
    std::cout << "  --input \"v1,v2,v3,...\"                Comma-separated feature vector for predict mode\n\n";
    std::cout << "General:\n";
    std::cout << "  -h, --help                            Display this help menu and exit\n";
}

CliArgs parseCommandLine(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            args.dataPath = argv[++i];
        } else if (arg == "--targets" && i + 1 < argc) {
            args.targetColumns = std::stoi(argv[++i]);
        } else if (arg == "--has-header") {
            args.hasHeader = true;
        } else if (arg == "--epochs" && i + 1 < argc) {
            args.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batchSize = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--lr" && i + 1 < argc) {
            args.lr = std::stod(argv[++i]);
        } else if (arg == "--weight-decay" && i + 1 < argc) {
            args.weightDecay = std::stod(argv[++i]);
        } else if (arg == "--optimizer" && i + 1 < argc) {
            args.optimizerName = argv[++i];
        } else if (arg == "--loss" && i + 1 < argc) {
            args.lossName = argv[++i];
        } else if (arg == "--test-ratio" && i + 1 < argc) {
            args.testRatio = std::stod(argv[++i]);
        } else if (arg == "--val-ratio" && i + 1 < argc) {
            args.valRatio = std::stod(argv[++i]);
        } else if (arg == "--patience" && i + 1 < argc) {
            args.patience = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--model-in" && i + 1 < argc) {
            args.modelIn = argv[++i];
        } else if (arg == "--model-out" && i + 1 < argc) {
            args.modelOut = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string item;
            args.predictInput.clear();
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) args.predictInput.push_back(std::stod(item));
            }
        } else {
            std::cerr << "Warning: Unknown argument '" << arg << "'. Use --help for usage.\n";
        }
    }
    return args;
}

std::unique_ptr<IOptimizer> createOptimizer(const std::string& name, double lr, double wd) {
    if (name == "adam") return std::make_unique<Adam>(lr, 0.9, 0.999, 1e-8, wd);
    if (name == "momentum") return std::make_unique<Momentum>(lr, 0.9, wd);
    if (name == "sgd") return std::make_unique<SGD>(lr, wd);
    if (name == "rmsprop") return std::make_unique<RMSprop>(lr, 0.99, 1e-8, wd);
    throw std::invalid_argument("Unknown optimizer: " + name);
}

std::unique_ptr<ILossFunction> createLoss(const std::string& name) {
    if (name == "bce") return std::make_unique<BinaryCrossEntropy>();
    if (name == "mse") return std::make_unique<MSE>();
    if (name == "cce") return std::make_unique<CategoricalCrossEntropy>();
    throw std::invalid_argument("Unknown loss: " + name);
}

int runTrainMode(const CliArgs& args) {
    std::cout << "========================================================\n";
    std::cout << "                 MiniANN CLI: Train Mode                \n";
    std::cout << "========================================================\n";

    Matrix X, y;
    std::cout << "Loading dataset from " << args.dataPath << "...\n";
    if (!DataLoader::loadCSV(args.dataPath, X, y, args.targetColumns, args.hasHeader)) {
        return 1;
    }
    std::cout << "Loaded " << X.rows() << " rows with " << X.cols() << " features and "
              << y.cols() << " target(s).\n\n";

    DatasetSplit split = DataLoader::trainTestSplit(X, y, args.testRatio, args.valRatio, args.seed);
    std::cout << "Split counts -> Train: " << split.X_train.rows()
              << " | Val: " << split.X_val.rows()
              << " | Test: " << split.X_test.rows() << "\n\n";

    auto opt = createOptimizer(args.optimizerName, args.lr, args.weightDecay);
    auto loss = createLoss(args.lossName);

    NeuralNetwork net(std::move(opt), std::move(loss), args.seed);
    const int inDim = static_cast<int>(split.X_train.cols());
    const int outDim = static_cast<int>(split.y_train.cols());

    // Construct balanced architecture
    net.addLayer(16, inDim, std::make_shared<Tanh>());
    net.addLayer(8, 16, std::make_shared<Tanh>());
    if (args.lossName == "cce" || outDim > 1) {
        net.addLayer(outDim, 8, std::make_shared<Softmax>());
    } else {
        net.addLayer(outDim, 8, std::make_shared<Sigmoid>());
    }

    TrainingConfig config;
    config.epochs = args.epochs;
    config.batchSize = args.batchSize;
    config.patience = args.patience;
    config.lrDecayStep = std::max(50, args.epochs / 3);
    config.lrDecayRate = 0.5;
    config.logInterval = std::max(1, args.epochs / 5);
    config.verbose = true;

    std::cout << "Starting training with " << args.optimizerName << "...\n";
    TrainingHistory history = net.train(split.X_train, split.y_train, split.X_val, split.y_val, config);

    std::cout << "\nTraining stopped at epoch " << history.stoppedEpoch
              << " (Early stopped: " << (history.earlyStopped ? "Yes" : "No") << ")\n\n";

    if (!split.X_test.empty()) {
        std::cout << ">>> Evaluating on Held-out Test Set (" << split.X_test.rows() << " samples)...\n";
        Matrix preds = net.predict(split.X_test);
        const double testLoss = net.computeLoss(preds, split.y_test) /
                                static_cast<double>(split.X_test.rows() * split.y_test.cols());
        std::cout << "Test Loss: " << std::fixed << std::setprecision(5) << testLoss << "\n";
        ClassificationMetrics metrics = Metrics::evaluate(preds, split.y_test, 0.5);
        Metrics::printReport(metrics, std::cout);
    }

    if (!net.saveModel(args.modelOut)) {
        return 1;
    }
    return 0;
}

int runEvalMode(const CliArgs& args) {
    std::cout << "========================================================\n";
    std::cout << "                 MiniANN CLI: Eval Mode                 \n";
    std::cout << "========================================================\n";

    NeuralNetwork net;
    std::cout << "Loading model from " << args.modelIn << "...\n";
    if (!net.loadModel(args.modelIn)) {
        return 1;
    }

    Matrix X, y;
    std::cout << "Loading evaluation dataset from " << args.dataPath << "...\n";
    if (!DataLoader::loadCSV(args.dataPath, X, y, args.targetColumns, args.hasHeader)) {
        return 1;
    }

    Matrix preds = net.predict(X);
    ClassificationMetrics metrics = Metrics::evaluate(preds, y, 0.5);
    Metrics::printReport(metrics, std::cout);
    return 0;
}

int runPredictMode(const CliArgs& args) {
    std::cout << "========================================================\n";
    std::cout << "               MiniANN CLI: Predict Mode                \n";
    std::cout << "========================================================\n";

    NeuralNetwork net;
    std::cout << "Loading model from " << args.modelIn << "...\n";
    if (!net.loadModel(args.modelIn)) {
        return 1;
    }

    std::cout << "Input features: [";
    for (std::size_t i = 0; i < args.predictInput.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << args.predictInput[i];
    }
    std::cout << "]\n\n";

    std::vector<double> out = net.predict(args.predictInput);
    std::cout << "Model Raw Output: [";
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << std::fixed << std::setprecision(5) << out[i];
    }
    std::cout << "]\n";

    if (out.size() == 1) {
        const int predictedClass = (out[0] >= 0.5) ? 1 : 0;
        std::cout << "Predicted Class: " << predictedClass
                  << " (Probability: " << (out[0] * 100.0) << "%)\n";
    } else {
        std::size_t bestC = 0;
        double maxProb = out[0];
        for (std::size_t c = 1; c < out.size(); ++c) {
            if (out[c] > maxProb) {
                maxProb = out[c];
                bestC = c;
            }
        }
        std::cout << "Predicted Class: " << bestC
                  << " (Probability: " << (maxProb * 100.0) << "%)\n";
    }
    return 0;
}

int runXorMode() {
    std::cout << "========================================================\n";
    std::cout << "                 MiniANN CLI: XOR Demo                  \n";
    std::cout << "========================================================\n";

    const Matrix X = Matrix::from2D({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}});
    const Matrix y = Matrix::from2D({{0.0}, {1.0}, {1.0}, {0.0}});

    NeuralNetwork net(
        std::make_unique<Momentum>(0.2, 0.9),
        std::make_unique<MSE>(),
        42);
    net.addLayer(4, 2, std::make_shared<Tanh>());
    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    std::cout << "Training XOR with Momentum optimizer for 2000 epochs...\n";
    net.train(X, y, 2000, 4);

    Matrix preds = net.predict(X);
    std::cout << "\nResults:\n";
    for (std::size_t i = 0; i < 4; ++i) {
        std::cout << "  Input: (" << X(i, 0) << ", " << X(i, 1)
                  << ") -> Expected: " << y(i, 0)
                  << " | Predicted: " << std::fixed << std::setprecision(5) << preds(i, 0)
                  << " (" << (preds(i, 0) >= 0.5 ? 1 : 0) << ")\n";
    }
    return 0;
}

} // namespace miniann

int main(int argc, char* argv[]) {
    try {
        miniann::CliArgs args = miniann::parseCommandLine(argc, argv);
        if (args.mode == "train") {
            return miniann::runTrainMode(args);
        } else if (args.mode == "eval") {
            return miniann::runEvalMode(args);
        } else if (args.mode == "predict") {
            return miniann::runPredictMode(args);
        } else if (args.mode == "xor") {
            return miniann::runXorMode();
        } else {
            std::cerr << "Error: Unknown mode '" << args.mode << "'. Valid modes: train, eval, predict, xor.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
}

