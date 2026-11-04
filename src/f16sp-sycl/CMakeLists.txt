# f16sp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16sp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
