# bezier-surface-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bezier-surface
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
