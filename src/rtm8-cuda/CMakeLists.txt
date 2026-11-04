# rtm8-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rtm8
    MODEL cuda
    SOURCES rtm8.cu
    CATEGORIES simulation
)
