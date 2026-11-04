# b+tree-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/b+tree/b+tree.tar.bz")

add_hecbench_benchmark(
    NAME b+tree
    MODEL cuda
    SOURCES main.cu
            kernel/kernel_wrapper.cu  
            kernel/kernel2_wrapper.cu  
            util/num/num.c
    CATEGORIES search
)
