# minibude-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME minibude
    MODEL hip
    SOURCES kernel.cu main.cu
    CATEGORIES bioinformatics
)
