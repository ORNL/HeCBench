# tensorT-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorT
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
