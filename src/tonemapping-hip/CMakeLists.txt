# tonemapping-hip/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/tonemapping/input.hdr")

add_hecbench_benchmark(
    NAME tonemapping
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
