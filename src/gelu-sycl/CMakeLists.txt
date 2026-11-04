# gelu-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gelu
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
