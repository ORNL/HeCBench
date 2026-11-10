# pitch-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pitch
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
