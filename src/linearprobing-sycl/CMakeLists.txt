# linearprobing-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME linearprobing
    MODEL sycl
    SOURCES linearprobing.cpp main.cpp test.cpp
    CATEGORIES search
)
