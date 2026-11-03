# heun-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME heun
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
    # Keep device multiply-adds unfused. The Heun update cancels O(sigma)
    # quantities, and nvcc's default FMA contraction does not match the host
    # reference rounding.
    COMPILE_OPTIONS -fmad=false
    LINK_LIBRARIES -lcudnn
    TEST_REGEX "PASS"
    TEST_ARGS 1 1 4 8 8 4
)
