# seam-carving-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME seam-carving
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
