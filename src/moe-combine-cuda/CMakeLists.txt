add_hecbench_benchmark(
    NAME moe-combine
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    TEST_ARGS --tokens 32 --hidden 128 --topk 8 --iters 2 --warmup 1
    TEST_REGEX "moe-combine: PASS"
)
