# is-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME is
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
