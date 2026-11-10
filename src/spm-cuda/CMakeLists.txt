# spm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME spm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
