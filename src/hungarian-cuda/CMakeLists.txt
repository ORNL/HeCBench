# hungarian-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hungarian
    MODEL cuda
    SOURCES main.cu
    CATEGORIES graph
)
