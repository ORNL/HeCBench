# reaction-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME reaction
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
