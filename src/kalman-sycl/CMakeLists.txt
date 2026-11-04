# kalman-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME kalman
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
