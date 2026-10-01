# muon-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME muon
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
