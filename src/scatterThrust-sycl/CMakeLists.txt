# scatterThrust-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterThrust
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
