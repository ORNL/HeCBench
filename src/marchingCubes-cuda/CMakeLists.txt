# marchingCubes-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME marchingCubes
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
