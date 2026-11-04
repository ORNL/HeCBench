# perlin-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME perlin
    MODEL sycl
    SOURCES main.cpp noise.cpp
    CATEGORIES imaging
)
