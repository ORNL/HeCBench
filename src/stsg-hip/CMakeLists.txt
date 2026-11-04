# stsg-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../stsg-cuda")

add_hecbench_benchmark(
    NAME stsg
    MODEL hip
    SOURCES main.cu ${INC_DIR}/Filter.cu
    CATEGORIES geoscience
    LINK_LIBRARIES gdal
)
