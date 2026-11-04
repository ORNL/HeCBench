# lombscargle-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lombscargle
    MODEL cuda
    SOURCES main.cu
    CATEGORIES signal
)
