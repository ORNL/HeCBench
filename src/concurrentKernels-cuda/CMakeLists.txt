# concurrentKernels-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME concurrentKernels
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
