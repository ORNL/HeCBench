# scatterAdd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterAdd
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
