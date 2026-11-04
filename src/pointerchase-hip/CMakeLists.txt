# pointerchase-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointerchase
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
