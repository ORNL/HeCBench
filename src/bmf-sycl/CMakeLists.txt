# bmf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bmf
    MODEL sycl
    SOURCES src/main.cpp
    CATEGORIES ml
)
