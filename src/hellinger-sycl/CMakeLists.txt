# hellinger-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hellinger
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
