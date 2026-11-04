# sc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sc
    MODEL cuda
    SOURCES device_sc.cu host_sc.cpp main.cu
    CATEGORIES compression
    LINK_LIBRARIES pthread
)
