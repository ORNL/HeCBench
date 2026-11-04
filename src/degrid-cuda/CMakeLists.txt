# degrid-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
    COMPILE_OPTIONS -DPRECISION=double
)
