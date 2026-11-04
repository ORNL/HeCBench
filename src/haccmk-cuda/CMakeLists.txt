# haccmk-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME haccmk
    MODEL cuda
    SOURCES haccmk.cu
    CATEGORIES simulation
)
