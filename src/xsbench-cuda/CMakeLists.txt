# xsbench-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME xsbench
    MODEL cuda
    SOURCES GridInit.cu Main.cu Materials.cu Simulation.cu XSutils.cu io.cu
    CATEGORIES simulation
)
