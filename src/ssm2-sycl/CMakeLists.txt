# ssm2-sycl/CMakeLists.txt
#
# The host reference uses OpenMP. FindSYCL picks icpx or clang++ from
# Intel LLVM, and both accept -Xarch_host -fopenmp. Each driver then links
# its own runtime. An extra -lomp makes icpx load a second runtime.

add_hecbench_benchmark(
    NAME ssm2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS "SHELL:-Xarch_host -fopenmp"
    LINK_LIBRARIES -Xarch_host -fopenmp
    TEST_REGEX "PASS"
    TEST_ARGS 1
)
