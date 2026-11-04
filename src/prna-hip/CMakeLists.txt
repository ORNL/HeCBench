# prna-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../prna-cuda")

add_hecbench_benchmark(
    NAME prna
    MODEL hip
    SOURCES prna.cu
            ${INC_DIR}/main.c
            ${INC_DIR}/base.c
            ${INC_DIR}/param.c
            ${INC_DIR}/util.c
    CATEGORIES bioinformatics
    COMPILE_OPTIONS -DFLOAT -ffast-math
)
