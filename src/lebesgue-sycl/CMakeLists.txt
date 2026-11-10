# lebesgue-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lebesgue
    MODEL sycl
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES math
)
