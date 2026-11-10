# gabor-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
    COMPILE_OPTIONS -ffast-math
)
