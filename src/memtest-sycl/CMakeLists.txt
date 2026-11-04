# memtest-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME memtest
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
