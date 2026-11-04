# lda-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lda
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
