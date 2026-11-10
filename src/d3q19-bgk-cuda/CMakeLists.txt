# d3q19-bgk-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME d3q19-bgk
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
