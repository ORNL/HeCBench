# deredundancy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL sycl
    SOURCES main.cpp utils.cpp
    CATEGORIES bioinformatics
)
