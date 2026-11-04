# particle-diffusion-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME particle-diffusion
    MODEL hip
    SOURCES motionsim.cu
    CATEGORIES simulation
)
