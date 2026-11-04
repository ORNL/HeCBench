# face-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../face-cuda")

add_hecbench_benchmark(
    NAME face
    MODEL hip
    SOURCES haar.cu
    ${INC_DIR}/image.cu
    ${INC_DIR}/main.cu
    ${INC_DIR}/rectangles.cu
    ${INC_DIR}/stdio-wrapper.cu
    CATEGORIES imaging
)
