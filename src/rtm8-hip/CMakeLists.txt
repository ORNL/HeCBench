# rtm8-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rtm8
    MODEL hip
    SOURCES rtm8.cu
    CATEGORIES simulation
)
