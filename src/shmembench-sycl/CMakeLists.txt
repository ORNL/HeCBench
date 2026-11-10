# shmembench-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME shmembench
    MODEL sycl
    SOURCES main.cpp shmem_kernels.cpp
    CATEGORIES bandwidth
)
