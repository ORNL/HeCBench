# dropout-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dropout
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
