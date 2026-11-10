# compute-score-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME compute-score
    MODEL sycl
    SOURCES main.cpp options.cpp
    CATEGORIES compression
)
