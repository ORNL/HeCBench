# binomial-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME binomial
    MODEL hip
    SOURCES kernel.cu main.cu reference.cu
    CATEGORIES finance
)
