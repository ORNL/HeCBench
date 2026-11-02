# attentionMergeState-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME attentionMergeState
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
