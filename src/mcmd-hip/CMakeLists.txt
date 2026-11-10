# mcmd-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../mcmd-cuda")

add_hecbench_benchmark(
    NAME mcmd
    MODEL hip
    SOURCES force_kernel.cu ${INC_DIR}/main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -DGPU
)
