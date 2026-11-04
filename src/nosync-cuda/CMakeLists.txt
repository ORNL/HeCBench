# nosync-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nosync
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
    LINK_LIBRARIES pthread
)
