add_hecbench_benchmark(
    NAME moe-combine
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    TEST_ARGS --tokens 32 --hidden 128 --topk 8 --iters 2 --warmup 1
    TEST_REGEX "moe-combine: PASS"
)
