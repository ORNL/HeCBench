# b+tree-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME b+tree
    MODEL hip
    SOURCES main.cu
            kernel/kernel_wrapper.cu  
            kernel/kernel2_wrapper.cu  
            util/num/num.c
    CATEGORIES search
)
