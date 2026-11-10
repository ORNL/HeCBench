# wsm5-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME wsm5
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -DMKX=4
)
