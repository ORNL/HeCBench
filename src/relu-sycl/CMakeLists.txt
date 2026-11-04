# relu-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME relu
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
