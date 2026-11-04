# haccmk-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME haccmk
    MODEL hip
    SOURCES haccmk.cu
    CATEGORIES simulation
)
