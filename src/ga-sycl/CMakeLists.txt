# ga-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ga
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
