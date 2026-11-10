# minmax-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minmax
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
