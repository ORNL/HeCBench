# marchingCubes-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME marchingCubes
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
