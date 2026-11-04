# rowwiseMoments-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rowwiseMoments
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
