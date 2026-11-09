# msm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME msm
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography
)
