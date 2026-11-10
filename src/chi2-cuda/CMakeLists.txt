# chi2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME chi2
    MODEL cuda
    SOURCES chi2.cu
    CATEGORIES simulation
)
