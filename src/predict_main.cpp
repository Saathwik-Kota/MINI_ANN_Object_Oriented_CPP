#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

int main() {
    using namespace miniann;

    NeuralNetwork myNet;

    std::cout << "Loading trained weights...\n";
    if (!myNet.loadModel("data/trained_classification_model.csv")) {
        return 1;
    }

    const std::vector<double> test_input = {0.8, 0.7, 0.6, 0.5};

    std::cout << "\nRunning prediction...\n";
    const std::vector<double> prediction = myNet.predict(test_input);

    std::cout << "Input Data: (" << test_input[0] << ", " << test_input[1] << ", "
              << test_input[2] << ", " << test_input[3] << ")\n";
    std::cout << "Network Output (Probability): " << prediction[0] << "\n";

    const int predicted_class = (prediction[0] >= 0.5) ? 1 : 0;
    std::cout << "Predicted Class: " << predicted_class << "\n";

    return 0;
}