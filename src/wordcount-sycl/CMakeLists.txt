# wordcount-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wordcount
    MODEL sycl
    SOURCES main.cpp wc.cpp
    CATEGORIES language
)
