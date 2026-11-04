# unfold-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME unfold
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
