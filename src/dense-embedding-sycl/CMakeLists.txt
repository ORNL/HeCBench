# dense-embedding-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dense-embedding
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
