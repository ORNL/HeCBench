# moe-sum-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-sum
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
