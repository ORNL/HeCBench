# moe-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
