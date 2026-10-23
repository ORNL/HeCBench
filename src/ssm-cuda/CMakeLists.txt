# ssm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)

# Suite default is C++17. This target is built as C++20.
if(TARGET ssm-cuda)
    target_compile_features(ssm-cuda PRIVATE cuda_std_20)
endif()
