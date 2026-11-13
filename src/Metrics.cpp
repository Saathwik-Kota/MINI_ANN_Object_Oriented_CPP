#include "Metrics.hpp"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace miniann {

ClassificationMetrics Metrics::evaluateBinary(const Matrix& predictions,
                                              const Matrix& targets,
                                              double threshold) {
    if (predictions.rows() != targets.rows() || predictions.empty()) {
        throw std::invalid_argument("Predictions and targets must be non-empty with equal row count.");
    }

    ClassificationMetrics m;
    m.isBinary = true;
    m.numClasses = 2;
    m.totalSamples = predictions.rows();
    m.confusionMatrix.assign(2, std::vector<std::size_t>(2, 0));
    m.classSupport.assign(2, 0);

    std::size_t tp = 0;
    std::size_t fp = 0;
    std::size_t tn = 0;
    std::size_t fn = 0;

    for (std::size_t i = 0; i < m.totalSamples; ++i) {
        const int actual = targets(i, 0) >= 0.5 ? 1 : 0;
        const int pred = predictions(i, 0) >= threshold ? 1 : 0;

        m.confusionMatrix[static_cast<std::size_t>(actual)][static_cast<std::size_t>(pred)]++;
        m.classSupport[static_cast<std::size_t>(actual)]++;

        if (actual == 1 && pred == 1) ++tp;
        else if (actual == 0 && pred == 1) ++fp;
        else if (actual == 0 && pred == 0) ++tn;
        else if (actual == 1 && pred == 0) ++fn;
    }

    m.accuracy = static_cast<double>(tp + tn) / static_cast<double>(m.totalSamples);
    m.precision = (tp + fp > 0) ? (static_cast<double>(tp) / static_cast<double>(tp + fp)) : 0.0;
    m.recall = (tp + fn > 0) ? (static_cast<double>(tp) / static_cast<double>(tp + fn)) : 0.0;
    m.f1Score = (m.precision + m.recall > 0.0)
                    ? (2.0 * m.precision * m.recall / (m.precision + m.recall))
                    : 0.0;

    // Class 0 metrics
    const double prec0 = (tn + fn > 0) ? (static_cast<double>(tn) / static_cast<double>(tn + fn)) : 0.0;
    const double rec0 = (tn + fp > 0) ? (static_cast<double>(tn) / static_cast<double>(tn + fp)) : 0.0;
    const double f1_0 = (prec0 + rec0 > 0.0) ? (2.0 * prec0 * rec0 / (prec0 + rec0)) : 0.0;

    m.classPrecision = {prec0, m.precision};
    m.classRecall = {rec0, m.recall};
    m.classF1 = {f1_0, m.f1Score};

    return m;
}

ClassificationMetrics Metrics::evaluateMultiClass(const Matrix& predictions,
                                                  const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.empty()) {
        throw std::invalid_argument("Predictions and targets must be non-empty with equal row count.");
    }

    const std::size_t totalSamples = predictions.rows();
    const std::size_t numClasses = predictions.cols();

    ClassificationMetrics m;
    m.isBinary = false;
    m.numClasses = static_cast<int>(numClasses);
    m.totalSamples = totalSamples;
    m.confusionMatrix.assign(numClasses, std::vector<std::size_t>(numClasses, 0));
    m.classSupport.assign(numClasses, 0);

    std::size_t correct = 0;

    for (std::size_t i = 0; i < totalSamples; ++i) {
        // Argmax prediction
        std::size_t predClass = 0;
        double maxPred = predictions(i, 0);
        for (std::size_t c = 1; c < numClasses; ++c) {
            if (predictions(i, c) > maxPred) {
                maxPred = predictions(i, c);
                predClass = c;
            }
        }

        // True class (from one-hot or index column)
        std::size_t trueClass = 0;
        if (targets.cols() == 1) {
            trueClass = static_cast<std::size_t>(targets(i, 0));
        } else {
            double maxTarget = targets(i, 0);
            for (std::size_t c = 1; c < targets.cols(); ++c) {
                if (targets(i, c) > maxTarget) {
                    maxTarget = targets(i, c);
                    trueClass = c;
                }
            }
        }

        if (trueClass < numClasses && predClass < numClasses) {
            m.confusionMatrix[trueClass][predClass]++;
            m.classSupport[trueClass]++;
            if (trueClass == predClass) {
                ++correct;
            }
        }
    }

    m.accuracy = static_cast<double>(correct) / static_cast<double>(totalSamples);
    m.classPrecision.assign(numClasses, 0.0);
    m.classRecall.assign(numClasses, 0.0);
    m.classF1.assign(numClasses, 0.0);

    double sumPrec = 0.0;
    double sumRec = 0.0;
    double sumF1 = 0.0;

    for (std::size_t c = 0; c < numClasses; ++c) {
        const std::size_t tp = m.confusionMatrix[c][c];
        std::size_t predictedC = 0;
        for (std::size_t r = 0; r < numClasses; ++r) {
            predictedC += m.confusionMatrix[r][c];
        }
        const std::size_t actualC = m.classSupport[c];

        const double p = (predictedC > 0) ? (static_cast<double>(tp) / static_cast<double>(predictedC)) : 0.0;
        const double r = (actualC > 0) ? (static_cast<double>(tp) / static_cast<double>(actualC)) : 0.0;
        const double f = (p + r > 0.0) ? (2.0 * p * r / (p + r)) : 0.0;

        m.classPrecision[c] = p;
        m.classRecall[c] = r;
        m.classF1[c] = f;

        sumPrec += p;
        sumRec += r;
        sumF1 += f;
    }

    m.precision = sumPrec / static_cast<double>(numClasses);
    m.recall = sumRec / static_cast<double>(numClasses);
    m.f1Score = sumF1 / static_cast<double>(numClasses);

    return m;
}

ClassificationMetrics Metrics::evaluate(const Matrix& predictions,
                                        const Matrix& targets,
                                        double threshold) {
    if (predictions.cols() == 1 && targets.cols() == 1) {
        return evaluateBinary(predictions, targets, threshold);
    }
    return evaluateMultiClass(predictions, targets);
}

double Metrics::accuracy(const Matrix& predictions,
                         const Matrix& targets,
                         double threshold) {
    return evaluate(predictions, targets, threshold).accuracy;
}

void Metrics::printReport(const ClassificationMetrics& m, std::ostream& os) {
    os << "--------------------------------------------------------\n";
    os << "              Classification Performance Report          \n";
    os << "--------------------------------------------------------\n";
    os << std::fixed << std::setprecision(4);
    os << " Overall Accuracy : " << (m.accuracy * 100.0) << "%\n";
    os << " Macro Precision  : " << m.precision << "\n";
    os << " Macro Recall     : " << m.recall << "\n";
    os << " Macro F1-Score   : " << m.f1Score << "\n";
    os << " Total Test Count : " << m.totalSamples << "\n\n";

    os << " Per-Class Breakdown:\n";
    os << "  Class | Precision |   Recall  |  F1-Score | Support\n";
    os << "  ------+-----------+-----------+-----------+--------\n";
    for (std::size_t c = 0; c < static_cast<std::size_t>(m.numClasses); ++c) {
        os << "  " << std::setw(5) << c << " | "
           << std::setw(9) << m.classPrecision[c] << " | "
           << std::setw(9) << m.classRecall[c] << " | "
           << std::setw(9) << m.classF1[c] << " | "
           << std::setw(7) << m.classSupport[c] << "\n";
    }

    os << "\n Confusion Matrix (Rows: Actual, Columns: Predicted):\n";
    os << "       ";
    for (std::size_t c = 0; c < static_cast<std::size_t>(m.numClasses); ++c) {
        os << std::setw(6) << ("P" + std::to_string(c));
    }
    os << "\n";
    for (std::size_t r = 0; r < static_cast<std::size_t>(m.numClasses); ++r) {
        os << "  A" << std::setw(3) << r << " [";
        for (std::size_t c = 0; c < static_cast<std::size_t>(m.numClasses); ++c) {
            os << std::setw(6) << m.confusionMatrix[r][c];
        }
        os << " ]\n";
    }
    os << "--------------------------------------------------------\n";
}

} // namespace miniann

