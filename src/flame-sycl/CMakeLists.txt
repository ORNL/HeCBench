# flame-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME flame
    MODEL sycl
    SOURCES main.cpp utils.cpp
    CATEGORIES imaging
)
