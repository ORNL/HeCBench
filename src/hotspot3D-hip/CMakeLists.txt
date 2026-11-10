# hotspot3D-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../hotspot3D-cuda")

add_hecbench_benchmark(
    NAME hotspot3D
    MODEL hip
    SOURCES 3D.cu ${INC_DIR}/3D_helper.cu
    CATEGORIES simulation
)
