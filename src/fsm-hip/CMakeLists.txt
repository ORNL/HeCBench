# fsm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fsm
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
