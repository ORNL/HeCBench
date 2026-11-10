# jaccard-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME jaccard
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
