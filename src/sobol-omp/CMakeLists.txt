# sobol-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobol
    MODEL omp
    SOURCES sobol.cpp sobol_gold.cpp sobol_gpu.cpp sobol_primitives.cpp
    CATEGORIES rng
)
