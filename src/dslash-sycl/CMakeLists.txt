# dslash-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dslash
    MODEL sycl
    SOURCES dslash.cpp kernels.cpp main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -ffast-math
)
