# groupnorm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME groupnorm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
