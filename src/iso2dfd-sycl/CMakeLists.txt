# iso2dfd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME iso2dfd
    MODEL sycl
    SOURCES iso2dfd.cpp
    CATEGORIES simulation
)
