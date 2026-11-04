# present-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME present
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
)
