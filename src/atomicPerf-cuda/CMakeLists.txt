# atomicPerf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicPerf
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
