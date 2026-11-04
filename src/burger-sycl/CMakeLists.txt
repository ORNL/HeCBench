# burger-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME burger
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
