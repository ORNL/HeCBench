# rtm8-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rtm8
    MODEL omp
    SOURCES rtm8.cpp
    CATEGORIES simulation
)
