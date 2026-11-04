# sa-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sa
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
