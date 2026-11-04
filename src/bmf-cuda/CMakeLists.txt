# bmf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bmf
    MODEL cuda
    SOURCES src/main.cu
    COMPILE_OPTIONS -Xcompiler=-fopenmp --default-stream=per-thread
    LINK_LIBRARIES -lgomp
    CATEGORIES ml
)
