# asyncAllocation-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME asyncAllocation
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language_features
)
