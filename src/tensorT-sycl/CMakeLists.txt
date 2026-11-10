# tensorT-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorT
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
