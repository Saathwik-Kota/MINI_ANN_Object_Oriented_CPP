#include "DataLoader.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace miniann {

namespace {

std::string trimWhitespace(const std::string& s) {
    std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

} // namespace

bool DataLoader::loadCSV(const std::string& filename,
                         std::vector<std::vector<double>>& X,
                         std::vector<std::vector<double>>& y,
                         int numTargets,
                         bool hasHeader) {
    if (numTargets <= 0) {
        std::cerr << "Error: numTargets must be positive.\n";
        return false;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return false;
    }

    X.clear();
    y.clear();

    std::string line;
    std::size_t lineNumber = 0;
    std::size_t expectedCols = 0;
    bool headerSkipped = !hasHeader;

    while (std::getline(file, line)) {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (trimWhitespace(line).empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            std::string trimmed = trimWhitespace(cell);
            if (trimmed.empty()) {
                std::cerr << "Error: Empty cell at line " << lineNumber << " in " << filename << "\n";
                X.clear();
                y.clear();
                return false;
            }
            try {
                std::size_t pos = 0;
                double val = std::stod(trimmed, &pos);
                if (pos != trimmed.size() || !std::isfinite(val)) {
                    throw std::invalid_argument("Trailing characters or non-finite value");
                }
                row.push_back(val);
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid numeric value '" << trimmed << "' at line "
                          << lineNumber << " in " << filename << "\n";
                X.clear();
                y.clear();
                return false;
            }
        }

        if (!line.empty() && line.back() == ',') {
            std::cerr << "Error: Trailing comma (empty cell) at line " << lineNumber
                      << " in " << filename << "\n";
            X.clear();
            y.clear();
            return false;
        }

        if (row.size() <= static_cast<std::size_t>(numTargets)) {
            std::cerr << "Error: Insufficient columns at line " << lineNumber << " in " << filename
                      << " (got " << row.size() << ", need > " << numTargets << ")\n";
            X.clear();
            y.clear();
            return false;
        }

        if (expectedCols == 0) {
            expectedCols = row.size();
        } else if (row.size() != expectedCols) {
            std::cerr << "Error: Inconsistent column count at line " << lineNumber << " in " << filename
                      << " (expected " << expectedCols << ", got " << row.size() << ")\n";
            X.clear();
            y.clear();
            return false;
        }

        std::vector<double> feature_row(row.begin(), row.end() - numTargets);
        std::vector<double> target_row(row.end() - numTargets, row.end());

        X.push_back(std::move(feature_row));
        y.push_back(std::move(target_row));
    }

    if (X.empty()) {
        std::cerr << "Error: No valid data rows found in " << filename << "\n";
        return false;
    }

    return true;
}

bool DataLoader::loadCSV(const std::string& filename,
                         Matrix& X,
                         Matrix& y,
                         int numTargets,
                         bool hasHeader) {
    std::vector<std::vector<double>> xRaw;
    std::vector<std::vector<double>> yRaw;
    if (!loadCSV(filename, xRaw, yRaw, numTargets, hasHeader)) {
        return false;
    }
    X = Matrix::from2D(xRaw);
    y = Matrix::from2D(yRaw);
    return true;
}

DatasetSplit DataLoader::trainTestSplit(const Matrix& X,
                                       const Matrix& y,
                                       double testRatio,
                                       double valRatio,
                                       unsigned int seed) {
    if (X.rows() != y.rows() || X.empty()) {
        throw std::invalid_argument("X and y must be non-empty and have matching row counts.");
    }
    if (testRatio < 0.0 || valRatio < 0.0 || (testRatio + valRatio) >= 1.0) {
        throw std::invalid_argument("Invalid test/val ratios; sum of testRatio and valRatio must be < 1.0.");
    }

    const std::size_t total = X.rows();
    const std::size_t inDim = X.cols();
    const std::size_t outDim = y.cols();

    const std::size_t nTest = static_cast<std::size_t>(std::round(static_cast<double>(total) * testRatio));
    const std::size_t nVal = static_cast<std::size_t>(std::round(static_cast<double>(total) * valRatio));
    const std::size_t nTrain = total - nTest - nVal;

    std::vector<std::size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    DatasetSplit split;
    split.X_train.resize(nTrain, inDim);
    split.y_train.resize(nTrain, outDim);
    for (std::size_t i = 0; i < nTrain; ++i) {
        split.X_train.setRow(i, X.rowPtr(indices[i]));
        split.y_train.setRow(i, y.rowPtr(indices[i]));
    }

    if (nVal > 0) {
        split.X_val.resize(nVal, inDim);
        split.y_val.resize(nVal, outDim);
        for (std::size_t i = 0; i < nVal; ++i) {
            split.X_val.setRow(i, X.rowPtr(indices[nTrain + i]));
            split.y_val.setRow(i, y.rowPtr(indices[nTrain + i]));
        }
    }

    if (nTest > 0) {
        split.X_test.resize(nTest, inDim);
        split.y_test.resize(nTest, outDim);
        for (std::size_t i = 0; i < nTest; ++i) {
            split.X_test.setRow(i, X.rowPtr(indices[nTrain + nVal + i]));
            split.y_test.setRow(i, y.rowPtr(indices[nTrain + nVal + i]));
        }
    }

    return split;
}

Matrix DataLoader::oneHotEncode(const Matrix& labels, std::size_t numClasses) {
    if (labels.cols() != 1 || numClasses == 0) {
        throw std::invalid_argument("oneHotEncode expects an Nx1 matrix of class indices and numClasses > 0.");
    }
    const std::size_t rows = labels.rows();
    Matrix encoded(rows, numClasses, 0.0);

    for (std::size_t r = 0; r < rows; ++r) {
        const double val = labels(r, 0);
        if (val < 0.0 || static_cast<std::size_t>(val) >= numClasses) {
            throw std::out_of_range("Label value " + std::to_string(val) + " out of range [0, " +
                                    std::to_string(numClasses - 1) + "].");
        }
        encoded(r, static_cast<std::size_t>(val)) = 1.0;
    }
    return encoded;
}

} // namespace miniann
