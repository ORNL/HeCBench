# binomial-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME binomial
    MODEL cuda
    SOURCES kernel.cu main.cu reference.cu
    CATEGORIES finance
)
