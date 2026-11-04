# svd3x3-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    COMPILE_OPTIONS -ffast-math
)
