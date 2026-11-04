# shuffle-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME shuffle
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
