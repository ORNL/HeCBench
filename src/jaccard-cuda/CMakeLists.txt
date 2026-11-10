# jaccard-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME jaccard
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
