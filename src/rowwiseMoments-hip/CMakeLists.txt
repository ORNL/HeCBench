# rowwiseMoments-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rowwiseMoments
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
