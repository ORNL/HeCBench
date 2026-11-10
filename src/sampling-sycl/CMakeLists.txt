# sampling-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
