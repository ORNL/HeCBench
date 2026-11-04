# phmm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME phmm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
