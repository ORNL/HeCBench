# bincount-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bincount
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
