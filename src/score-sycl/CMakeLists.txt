# score-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME score
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
