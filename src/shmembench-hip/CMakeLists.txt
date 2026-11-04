# shmembench-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME shmembench
    MODEL hip
    SOURCES main.cu shmem_kernels.cu
    CATEGORIES bandwidth
)
