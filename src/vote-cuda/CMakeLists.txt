# vote-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME vote
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
