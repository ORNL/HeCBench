# vmc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME vmc
    MODEL sycl
    SOURCES vmc2.cpp
    CATEGORIES simulation
)
