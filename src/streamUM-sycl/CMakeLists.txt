# streamUM-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
    COMPILE_OPTIONS -fopenmp 
    LINK_LIBRARIES -qmkl=parallel -fopenmp
)
