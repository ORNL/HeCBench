# popcount-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME popcount
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
