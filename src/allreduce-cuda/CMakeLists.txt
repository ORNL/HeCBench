# allreduce-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME allreduce
    MODEL cuda
    SOURCES collectives.cu main.cu timer.cu
    CATEGORIES bandwidth
)
