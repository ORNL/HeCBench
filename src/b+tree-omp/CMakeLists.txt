# b+tree-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME b+tree
    MODEL omp
    SOURCES main.c
            kernel/kernel_wrapper.c
            kernel/kernel2_wrapper.c
            util/num/num.c
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/util/num
                 ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    CATEGORIES algorithms
    LINK_LIBRARIES m
    C_AS_CXX
)
