# rtm8-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rtm8
    MODEL sycl
    SOURCES rtm8.cpp
    CATEGORIES simulation
)
