# romberg-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME romberg
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
