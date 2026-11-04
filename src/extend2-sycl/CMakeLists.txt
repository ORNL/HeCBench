# extend2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME extend2
    MODEL sycl
    SOURCES main.cpp read_data.cpp
    CATEGORIES bioinformatics
)
