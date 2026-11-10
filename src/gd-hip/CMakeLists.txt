# gd-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../gd-cuda")

add_hecbench_benchmark(
    NAME gd
    MODEL hip
    SOURCES main.cu ${INC_DIR}/utils.cpp
    COMPILE_OPTIONS -munsafe-fp-atomics
    CATEGORIES ml
)
