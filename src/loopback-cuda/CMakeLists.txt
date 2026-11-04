# loopback-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
