# f16atomic-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16atomic
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
