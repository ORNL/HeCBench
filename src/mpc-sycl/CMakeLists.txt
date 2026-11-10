# mpc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mpc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
