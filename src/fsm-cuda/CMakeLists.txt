# fsm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fsm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
