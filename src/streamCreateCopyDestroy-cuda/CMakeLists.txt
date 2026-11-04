# streamCreateCopyDestroy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamCreateCopyDestroy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
