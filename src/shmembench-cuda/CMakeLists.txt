# shmembench-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME shmembench
    MODEL cuda
    SOURCES main.cu shmem_kernels.cu
    CATEGORIES bandwidth
)
