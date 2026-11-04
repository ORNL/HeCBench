# haversine-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../haversine-cuda")

add_hecbench_benchmark(
    NAME haversine
    MODEL hip
    SOURCES ${INC_DIR}/main.cu distance.cu
    CATEGORIES geoscience
    COMPILE_OPTIONS -ffast-math
)
