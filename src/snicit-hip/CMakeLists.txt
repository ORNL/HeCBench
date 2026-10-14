# snicit-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME snicit
    MODEL hip
    SOURCES src/beyond.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../snicit-cuda/3rd-party
    CATEGORIES neuralnetwork algorithms
)
