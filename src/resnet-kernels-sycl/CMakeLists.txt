# resnet-kernels-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME resnet-kernels
    MODEL sycl
    SOURCES Kernel128_one.cpp Kernel128_winograd.cpp Kernel256_one.cpp Kernel256_winograd.cpp main.cpp util.cpp
    CATEGORIES ml
)
