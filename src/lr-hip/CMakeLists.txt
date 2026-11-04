# lr-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../lr-cuda")

add_hecbench_benchmark(
    NAME lr
    MODEL hip
    SOURCES linear_par.cu ${INC_DIR}/linear_iter.cu  ${INC_DIR}/main.cu 
    CATEGORIES ml
)
