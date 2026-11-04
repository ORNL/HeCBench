# mxfp4-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mxfp4
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
