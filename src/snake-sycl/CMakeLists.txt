# snake-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME snake
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
