# nbody-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbody
    MODEL hip
    SOURCES GSimulation.cu main.cu
    CATEGORIES simulation
)
