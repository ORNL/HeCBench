# eigenvalue-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME eigenvalue
    MODEL sycl
    SOURCES main.cpp reference.cpp
    CATEGORIES math
)
