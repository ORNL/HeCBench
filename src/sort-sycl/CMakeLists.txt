# sort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sort
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
