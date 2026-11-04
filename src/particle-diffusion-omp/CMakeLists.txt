# particle-diffusion-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME particle-diffusion
    MODEL omp
    SOURCES motionsim.cpp
    CATEGORIES simulation
)
