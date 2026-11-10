# ntt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ntt
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    COMPILE_OPTIONS -Wno-register
)
