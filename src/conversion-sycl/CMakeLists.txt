# conversion-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME conversion
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
