# bh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bh
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
