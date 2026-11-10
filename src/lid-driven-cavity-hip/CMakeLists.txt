# lid-driven-cavity-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lid-driven-cavity
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
