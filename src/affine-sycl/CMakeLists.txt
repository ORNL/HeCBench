# affine-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME affine
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
