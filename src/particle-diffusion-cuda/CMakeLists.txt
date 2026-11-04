# particle-diffusion-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME particle-diffusion
    MODEL cuda
    SOURCES motionsim.cu
    CATEGORIES simulation
)
