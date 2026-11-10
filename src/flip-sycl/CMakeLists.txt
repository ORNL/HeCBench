# flip-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME flip
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
