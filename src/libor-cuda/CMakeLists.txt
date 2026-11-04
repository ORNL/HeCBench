# libor-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME libor
    MODEL cuda
    SOURCES main.cu
    CATEGORIES finance
)
