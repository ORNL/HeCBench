# addBiasQKV-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME addBiasQKV
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
