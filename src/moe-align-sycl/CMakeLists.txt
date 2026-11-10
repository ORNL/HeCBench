# moe-align-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-align
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
