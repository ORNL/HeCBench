# mcmd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcmd
    MODEL cuda
    SOURCES main.cpp force_kernel.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -DGPU
)
