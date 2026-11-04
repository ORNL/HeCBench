# adv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -Ddfloat=double -Ddlong=int
)
