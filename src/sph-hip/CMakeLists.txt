# sph-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sph
    MODEL hip
    SOURCES fileio.cu fluid.cu geometry.cu
    CATEGORIES simulation
)
