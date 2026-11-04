# idivide-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME idivide
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
