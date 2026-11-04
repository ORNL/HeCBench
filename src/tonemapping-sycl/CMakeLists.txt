# tonemapping-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
