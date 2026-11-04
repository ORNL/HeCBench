# sobol-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobol
    MODEL hip
    SOURCES sobol.cu sobol_gold.cu sobol_gpu.cu sobol_primitives.cu
    CATEGORIES rng
)
