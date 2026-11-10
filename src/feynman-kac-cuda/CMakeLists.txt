# feynman-kac-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME feynman-kac
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
