# msm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME msm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
)
