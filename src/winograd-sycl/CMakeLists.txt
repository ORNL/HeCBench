# winograd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME winograd
    MODEL sycl
    SOURCES main.cpp utils.cpp
    CATEGORIES ml
)
