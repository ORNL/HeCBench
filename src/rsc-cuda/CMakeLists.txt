# rsc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsc
    MODEL cuda
    SOURCES main.cu model_eval.cu model_fitting.cpp
    CATEGORIES math
    LINK_LIBRARIES pthread
)
