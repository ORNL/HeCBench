# doh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME doh
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
