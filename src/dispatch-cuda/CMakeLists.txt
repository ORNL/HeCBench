# dispatch-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dispatch
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
