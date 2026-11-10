# ne-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ne
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
