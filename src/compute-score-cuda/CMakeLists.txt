# compute-score-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME compute-score
    MODEL cuda
    SOURCES main.cu options.cu
    CATEGORIES compression
)
