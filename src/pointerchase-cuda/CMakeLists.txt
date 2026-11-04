# pointerchase-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointerchase
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
