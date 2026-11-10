# jacobi-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobi
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
