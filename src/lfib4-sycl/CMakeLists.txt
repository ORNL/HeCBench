# lfib4-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lfib4
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
