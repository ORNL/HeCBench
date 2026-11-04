# nbody-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbody
    MODEL sycl
    SOURCES GSimulation.cpp main.cpp
    CATEGORIES simulation
)
