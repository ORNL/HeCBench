# threadfence-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadfence
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
