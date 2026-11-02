# twell-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME twell
    MODEL cuda
    SOURCES
        main.cu
        twell_modules/matmul_d2t.cu
        twell_modules/matmul_t2d.cu
        twell_modules/matmul_gated_t2d.cu
        twell_modules/mlp_twell.cu
    COMPILE_OPTIONS
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --generate-code=arch=compute_90a,code=sm_90a
    LINK_LIBRARIES CUDA::cublas CUDA::cuda
    CATEGORIES ml
    TEST_ARGS --reps 100
)

# The kernels require the sm_90a feature set. Drop the suite-wide
# architecture so an older default (sm_80) is not compiled as well.
if(TARGET twell-cuda)
    set_target_properties(twell-cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_ARCHITECTURES OFF
    )
endif()
