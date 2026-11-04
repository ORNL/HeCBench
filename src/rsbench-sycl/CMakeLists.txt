# rsbench-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsbench
    MODEL sycl
    SOURCES init.cpp io.cpp main.cpp material.cpp simulation.cpp utils.cpp
    CATEGORIES simulation
)
