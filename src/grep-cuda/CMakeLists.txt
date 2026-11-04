# grep-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME grep
    MODEL cuda
    SOURCES main.cu nfautil.cu regex.cu
    CATEGORIES search
)
