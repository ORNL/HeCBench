# particlefilter-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME particlefilter
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
