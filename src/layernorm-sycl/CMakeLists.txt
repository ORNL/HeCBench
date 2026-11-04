# layernorm-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../rmsnorm-sycl")

add_hecbench_benchmark(
    NAME layernorm
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES ml
)
