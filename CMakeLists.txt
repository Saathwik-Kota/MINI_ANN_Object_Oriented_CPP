cmake_minimum_required(VERSION 3.16)
project(MiniANN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MINIANN_BUILD_TESTS "Build MiniANN unit and integration tests" ON)
option(MINIANN_BUILD_BENCHMARKS "Build MiniANN performance benchmarks" ON)
option(MINIANN_BUILD_EXAMPLES "Build MiniANN usage examples" ON)
option(MINIANN_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MINIANN_ENABLE_OPENMP "Enable OpenMP support if available" ON)

add_library(miniann_core STATIC
    src/DataLoader.cpp
    src/Layer.cpp
    src/Matrix.cpp
    src/Metrics.cpp
    src/NeuralNetwork.cpp
    src/Neuron.cpp
)

target_include_directories(miniann_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(miniann_core PUBLIC /W4 /WX)
else()
    target_compile_options(miniann_core PUBLIC -Wall -Wextra -Wpedantic -Werror)
endif()

if(MINIANN_ENABLE_SANITIZERS)
    if(MSVC)
        message(WARNING "MINIANN_ENABLE_SANITIZERS is only configured for GCC/Clang.")
    else()
        target_compile_options(miniann_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
        target_link_options(miniann_core PUBLIC -fsanitize=address,undefined)
    endif()
endif()

if(MINIANN_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(miniann_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Primary Unified CLI Executable
add_executable(miniann_cli src/cli_main.cpp)
target_link_libraries(miniann_cli PRIVATE miniann_core)

# Legacy pipeline executables (kept for backward-compatible workflow)
add_executable(miniann src/main.cpp)
target_link_libraries(miniann PRIVATE miniann_core)

add_executable(predict_miniann src/predict_main.cpp)
target_link_libraries(predict_miniann PRIVATE miniann_core)

add_executable(xor_miniann src/xor_main.cpp)
target_link_libraries(xor_miniann PRIVATE miniann_core)

# Tests
if(MINIANN_BUILD_TESTS)
    enable_testing()
    add_executable(miniann_tests tests/test_miniann.cpp)
    target_link_libraries(miniann_tests PRIVATE miniann_core)
    add_test(NAME miniann_unit_tests COMMAND miniann_tests)
endif()

# Benchmarks
if(MINIANN_BUILD_BENCHMARKS)
    add_executable(miniann_benchmarks benchmarks/benchmark_main.cpp)
    target_link_libraries(miniann_benchmarks PRIVATE miniann_core)

    add_executable(multiclass_benchmark benchmarks/multiclass_benchmark.cpp)
    target_link_libraries(multiclass_benchmark PRIVATE miniann_core)
endif()

# Examples
if(MINIANN_BUILD_EXAMPLES)
    add_executable(xor_example examples/xor_example.cpp)
    target_link_libraries(xor_example PRIVATE miniann_core)

    add_executable(custom_training_example examples/custom_training_example.cpp)
    target_link_libraries(custom_training_example PRIVATE miniann_core)
endif()
