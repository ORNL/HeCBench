# match-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME match
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
