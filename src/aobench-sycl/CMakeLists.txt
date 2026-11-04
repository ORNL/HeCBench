# aobench-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME aobench
    MODEL sycl
    SOURCES ao.cpp
    CATEGORIES imaging
    COMPILE_OPTIONS -ffast-math
)
