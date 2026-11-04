# bm3d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bm3d
    MODEL hip
    SOURCES blockmatching.cu dct8x8.cu filtering.cu main.cu
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
