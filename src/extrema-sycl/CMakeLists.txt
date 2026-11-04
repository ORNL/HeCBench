# extrema-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME extrema
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES signal
)
