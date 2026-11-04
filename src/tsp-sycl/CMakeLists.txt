# tsp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
    COMPILE_OPTIONS -ffast-math
)
