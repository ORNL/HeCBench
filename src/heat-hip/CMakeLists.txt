# heat-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat
    MODEL hip
    SOURCES heat.cu
    CATEGORIES simulation
)
