# mt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mt
    MODEL hip
    SOURCES MT.cu MT_gold.cu genmtrand.cu
    CATEGORIES rng
)
