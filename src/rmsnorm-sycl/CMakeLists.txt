# rmsnorm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
