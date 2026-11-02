# stiff-ode-sycl/CMakeLists.txt
#
# The host reference uses OpenMP. FindSYCL picks icpx or clang++ from
# Intel LLVM, and both accept -Xarch_host -fopenmp. add_hecbench_benchmark
# puts only SYCL_LINK_FLAGS on the link line, so the OpenMP runtime has to
# be requested here. -fopenmp lets icpx select libiomp5 and clang++ select
# libomp. An extra -lomp makes icpx load a second runtime.

add_hecbench_benchmark(
    NAME stiff-ode
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS "SHELL:-Xarch_host -fopenmp"
    LINK_LIBRARIES -fopenmp
    TEST_ARGS --zones 2 --iters 1 --steps 2 --warmup 0
    TEST_REGEX "stiff-ode: PASS"
    TEST_TIMEOUT 300
)
