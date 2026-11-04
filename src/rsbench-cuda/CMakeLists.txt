# rsbench-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsbench
    MODEL cuda
    SOURCES init.cu io.cu main.cu material.cu simulation.cu utils.cu
    CATEGORIES simulation
)
