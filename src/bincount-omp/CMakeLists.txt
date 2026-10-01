# bincount-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bincount
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)

# Exact histogram validation requires the host reference and Intel GPU kernel
# to use the same floating-point division semantics.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(bincount-omp PRIVATE -fp-model=precise)
    target_compile_definitions(bincount-omp PRIVATE
        __STRICT_ANSI__
    )
    target_link_options(bincount-omp PRIVATE
        "SHELL:-Xopenmp-target-backend -cl-fp32-correctly-rounded-divide-sqrt"
    )
endif()
