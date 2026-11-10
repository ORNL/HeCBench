# coordinates-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME coordinates
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES geoscience
)
