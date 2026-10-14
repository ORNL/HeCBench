# twell-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME twell
    MODEL hip
    SOURCES
        main.cu
        twell_modules/matmul_d2t.cu
        twell_modules/matmul_t2d.cu
        twell_modules/matmul_gated_t2d.cu
        twell_modules/mlp_twell.cu
    CATEGORIES ml
    TEST_ARGS --reps 100
)
