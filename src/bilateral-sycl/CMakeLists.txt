# bilateral-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bilateral
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
