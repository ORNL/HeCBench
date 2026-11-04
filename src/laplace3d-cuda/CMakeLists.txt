# laplace3d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace3d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
