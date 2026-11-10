# reverse-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
