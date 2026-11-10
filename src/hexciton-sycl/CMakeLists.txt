# hexciton-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hexciton
    MODEL sycl
    SOURCES main.cpp reference.cpp utils.cpp
    CATEGORIES simulation
)
