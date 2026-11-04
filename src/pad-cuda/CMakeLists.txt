# pad-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pad
    MODEL cuda
    SOURCES host_task.cpp kernel.cu main.cu
    CATEGORIES language
    LINK_LIBRARIES pthread
)
