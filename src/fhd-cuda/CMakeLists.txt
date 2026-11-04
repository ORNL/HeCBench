# fhd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fhd
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
