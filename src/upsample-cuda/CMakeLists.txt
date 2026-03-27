# upsample-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME upsample
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
