# sobol-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobol
    MODEL sycl
    SOURCES sobol.cpp sobol_gold.cpp sobol_gpu.cpp sobol_primitives.cpp
    CATEGORIES rng
)
