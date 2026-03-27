# upsample-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME upsample
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
