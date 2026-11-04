# crs-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../crs-cuda")

add_hecbench_benchmark(
    NAME crs
    MODEL hip
    SOURCES main.cu utils.cu
            ${INC_DIR}/galois.cu 
            ${INC_DIR}/jerasure.cu
            ${INC_DIR}/GCRSMatrix.cu
    CATEGORIES encoding
)
