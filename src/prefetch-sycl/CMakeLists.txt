# prefetch-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME prefetch
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
