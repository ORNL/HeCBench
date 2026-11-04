# ccl-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccl
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
    LINK_LIBRARIES nccl
)
