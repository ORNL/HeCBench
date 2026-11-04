# atomicAggregate-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicAggregate
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
