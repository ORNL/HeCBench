# accuracy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME accuracy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
