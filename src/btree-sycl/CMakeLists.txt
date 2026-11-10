# btree-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME btree
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
)
