# silu-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME silu
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
