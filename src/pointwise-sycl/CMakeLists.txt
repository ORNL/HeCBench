# pointwise-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointwise
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
