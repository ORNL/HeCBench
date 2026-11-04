# overlap-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME overlap
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
