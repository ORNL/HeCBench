# permute-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME permute
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
