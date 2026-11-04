# grrt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -ffast-math
)
