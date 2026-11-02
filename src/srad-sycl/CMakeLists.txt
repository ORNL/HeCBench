# srad-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL sycl
    SOURCES main.cpp
            util/graphics/graphics.c
            util/graphics/resize.c
    CATEGORIES algorithms
    C_AS_CXX
)
