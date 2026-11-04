# wlcpow-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wlcpow
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
