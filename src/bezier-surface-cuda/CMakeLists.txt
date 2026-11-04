# bezier-surface-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bezier-surface
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
