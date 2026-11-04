# geglu-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME geglu
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
