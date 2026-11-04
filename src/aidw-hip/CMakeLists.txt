# aidw-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME aidw
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES geoscience
)
