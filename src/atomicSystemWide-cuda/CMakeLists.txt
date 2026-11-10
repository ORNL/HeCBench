# atomicSystemWide-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicSystemWide
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
