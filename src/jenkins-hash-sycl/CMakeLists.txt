# jenkins-hash-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME jenkins-hash
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES encoding
)
