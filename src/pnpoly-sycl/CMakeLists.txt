# pnpoly-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pnpoly
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
