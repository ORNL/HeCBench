# ising-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ising
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
