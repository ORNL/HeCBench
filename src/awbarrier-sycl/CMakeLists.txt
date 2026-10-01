# awbarrier-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME awbarrier
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
