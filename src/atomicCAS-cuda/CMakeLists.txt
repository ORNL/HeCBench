# atomicCAS-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCAS
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
