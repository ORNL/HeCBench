# d2q9-bgk-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME d2q9-bgk
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
