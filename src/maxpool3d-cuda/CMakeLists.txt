# maxpool3d-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxpool3d
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
