# jenkins-hash-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME jenkins-hash
    MODEL cuda
    SOURCES main.cu
    CATEGORIES encoding
)
