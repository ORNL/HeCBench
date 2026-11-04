# ising-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ising
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
