# streamCreateCopyDestroy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamCreateCopyDestroy
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
