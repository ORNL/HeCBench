# bspline-vgh-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bspline-vgh
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
