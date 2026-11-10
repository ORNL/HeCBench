# iso2dfd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME iso2dfd
    MODEL cuda
    SOURCES iso2dfd.cu
    CATEGORIES simulation
)
