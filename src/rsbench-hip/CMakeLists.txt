# rsbench-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsbench
    MODEL hip
    SOURCES init.cu io.cu main.cu material.cu simulation.cu utils.cu
    CATEGORIES simulation
)
