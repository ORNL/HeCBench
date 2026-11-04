# scatter-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatter
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
