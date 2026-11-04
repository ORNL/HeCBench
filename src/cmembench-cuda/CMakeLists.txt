# cmembench-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cmembench
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
