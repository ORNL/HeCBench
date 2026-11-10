# spm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME spm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
