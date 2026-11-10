# quicksort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME quicksort
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
