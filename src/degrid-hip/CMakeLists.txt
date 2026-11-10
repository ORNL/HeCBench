# degrid-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
    COMPILE_OPTIONS -DPRECISION=double
)
