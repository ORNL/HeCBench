# f16max-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16max
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
