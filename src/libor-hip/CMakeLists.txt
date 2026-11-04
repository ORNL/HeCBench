# libor-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME libor
    MODEL hip
    SOURCES main.cu
    CATEGORIES finance
)
