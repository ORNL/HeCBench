# stencil3d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil3d
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
