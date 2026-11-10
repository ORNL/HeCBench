# rotary-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rotary
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    COMPILE_OPTIONS --extended-lambda --expt-relaxed-constexpr
)
