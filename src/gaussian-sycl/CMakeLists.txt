# gaussian-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gaussian
    MODEL sycl
    SOURCES gaussianElim.cpp utils.cpp
    CATEGORIES math
)
