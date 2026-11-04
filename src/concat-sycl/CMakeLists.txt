# concat-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME concat
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
