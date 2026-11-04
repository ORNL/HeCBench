# bh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bh
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
