# shmembench-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME shmembench
    MODEL omp
    SOURCES main.cpp shmem_kernels.cpp
    CATEGORIES bandwidth
)
