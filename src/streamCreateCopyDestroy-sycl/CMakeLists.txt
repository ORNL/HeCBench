# streamCreateCopyDestroy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamCreateCopyDestroy
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
