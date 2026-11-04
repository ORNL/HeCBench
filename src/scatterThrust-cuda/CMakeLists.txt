# scatterThrust-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterThrust
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
