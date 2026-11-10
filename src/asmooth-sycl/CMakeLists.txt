# asmooth-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME asmooth
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
