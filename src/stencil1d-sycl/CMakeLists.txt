# stencil1d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil1d
    MODEL sycl
    SOURCES stencil_1d.cpp
    CATEGORIES imaging
)
