# nbody-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbody
    MODEL cuda
    SOURCES GSimulation.cu main.cu
    CATEGORIES simulation
)
