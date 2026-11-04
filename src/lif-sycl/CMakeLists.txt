# lif-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lif
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
