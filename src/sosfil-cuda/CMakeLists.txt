# sosfil-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sosfil
    MODEL cuda
    SOURCES main.cu
    CATEGORIES signal
)
