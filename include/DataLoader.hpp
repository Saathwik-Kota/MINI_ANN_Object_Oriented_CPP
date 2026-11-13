#pragma once
#include <string>
#include <vector>
#include "Matrix.hpp"

namespace miniann {

struct DatasetSplit {
    Matrix X_train;
    Matrix y_train;
    Matrix X_val;
    Matrix y_val;
    Matrix X_test;
    Matrix y_test;
};

class DataLoader {
public:
    static bool loadCSV(const std::string& filename,
                        std::vector<std::vector<double>>& X,
                        std::vector<std::vector<double>>& y,
                        int numTargets,
                        bool hasHeader = false);

    static bool loadCSV(const std::string& filename,
                        Matrix& X,
                        Matrix& y,
                        int numTargets,
                        bool hasHeader = false);

    static DatasetSplit trainTestSplit(const Matrix& X,
                                       const Matrix& y,
                                       double testRatio = 0.2,
                                       double valRatio = 0.0,
                                       unsigned int seed = 42);

    static Matrix oneHotEncode(const Matrix& labels, std::size_t numClasses);
};

} // namespace miniann
