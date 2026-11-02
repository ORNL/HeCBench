# snicit-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME snicit
    MODEL sycl
    SOURCES src/beyond.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../snicit-cuda/3rd-party
    CATEGORIES neuralnetwork algorithms
)
