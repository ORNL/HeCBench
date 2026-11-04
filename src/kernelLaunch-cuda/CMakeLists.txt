# kernelLaunch-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kernelLaunch
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
