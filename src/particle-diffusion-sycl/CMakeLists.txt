# particle-diffusion-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME particle-diffusion
    MODEL sycl
    SOURCES motionsim.cpp
    CATEGORIES simulation
)
