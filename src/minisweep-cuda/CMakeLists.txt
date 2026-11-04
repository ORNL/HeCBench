# minisweep-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
