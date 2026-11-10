# mt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mt
    MODEL cuda
    SOURCES MT.cu MT_gold.cu genmtrand.cu
    CATEGORIES rng
)
