# frna-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../frna-cuda")

add_hecbench_benchmark(
    NAME frna
    MODEL hip
    SOURCES frna.cu
            ${INC_DIR}/main.c
            ${INC_DIR}/fbase.c
            ${INC_DIR}/fparam.c
            ${INC_DIR}/util.c
    CATEGORIES bioinformatics
    COMPILE_OPTIONS -DINT -ffast-math
)
