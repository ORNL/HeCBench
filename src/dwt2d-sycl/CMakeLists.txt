# dwt2d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dwt2d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
