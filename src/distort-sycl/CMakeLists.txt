# distort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME distort
    MODEL sycl
    SOURCES distort.cpp main.cpp
    CATEGORIES imaging
)
