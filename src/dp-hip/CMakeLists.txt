# dp-hip/CMakeLists.txt
#--hipstdpar-path=/opt/rocm/include/thrust/system/hip/hipstdpar

add_hecbench_benchmark(
    NAME dp
    MODEL hip
    SOURCES main.cu shrUtils.cu
    CATEGORIES math
    COMPILE_OPTIONS --hipstdpar 
                    -munsafe-fp-atomics
    LINK_LIBRARIES hipblas
)
