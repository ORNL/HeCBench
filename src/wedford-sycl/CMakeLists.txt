# wedford-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wedford
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
