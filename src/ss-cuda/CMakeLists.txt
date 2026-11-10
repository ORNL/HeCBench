# ss-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ss
    MODEL cuda
    SOURCES main.cu
    CATEGORIES search
)
