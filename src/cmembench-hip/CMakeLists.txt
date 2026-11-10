# cmembench-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cmembench
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
