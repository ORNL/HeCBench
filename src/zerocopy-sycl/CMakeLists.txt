# zerocopy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME zerocopy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
