# hwt1d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hwt1d
    MODEL sycl
    SOURCES kernel.cpp main.cpp reference.cpp
    CATEGORIES simulation
)
