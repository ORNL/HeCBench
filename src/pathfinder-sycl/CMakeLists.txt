# pathfinder-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pathfinder
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
