# heat2d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat2d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
