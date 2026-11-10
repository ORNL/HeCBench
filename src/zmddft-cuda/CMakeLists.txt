# zmddft-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME zmddft
    MODEL cuda
    SOURCES main.cu
    CATEGORIES signal
)
