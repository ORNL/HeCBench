# nbody-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbody
    MODEL omp
    SOURCES GSimulation.cpp main.cpp
    CATEGORIES simulation
)
