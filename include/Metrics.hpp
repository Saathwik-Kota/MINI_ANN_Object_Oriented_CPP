#pragma once
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include "Matrix.hpp"

namespace miniann {

struct ClassificationMetrics {
    bool isBinary{true};
    int numClasses{2};
    std::size_t totalSamples{0};

    double accuracy{0.0};
    double precision{0.0}; // Macro for multi-class; positive-class for binary
    double recall{0.0};
    double f1Score{0.0};

    std::vector<double> classPrecision;
    std::vector<double> classRecall;
    std::vector<double> classF1;
    std::vector<std::size_t> classSupport;

    // confusionMatrix[trueClass][predClass]
    std::vector<std::vector<std::size_t>> confusionMatrix;
};

class Metrics {
public:
    static ClassificationMetrics evaluateBinary(const Matrix& predictions,
                                                const Matrix& targets,
                                                double threshold = 0.5);

    static ClassificationMetrics evaluateMultiClass(const Matrix& predictions,
                                                    const Matrix& targets);

    static ClassificationMetrics evaluate(const Matrix& predictions,
                                          const Matrix& targets,
                                          double threshold = 0.5);

    static double accuracy(const Matrix& predictions,
                           const Matrix& targets,
                           double threshold = 0.5);

    static void printReport(const ClassificationMetrics& metrics,
                            std::ostream& os = std::cout);
};

} // namespace miniann

