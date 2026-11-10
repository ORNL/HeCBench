# heat-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat
    MODEL sycl
    SOURCES heat.cpp
    CATEGORIES simulation
)
