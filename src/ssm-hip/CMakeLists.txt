# ssm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ssm
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)

# Suite default is C++17, and HIP targets also set HIP_STANDARD 17.
# Request hip_std_20 after that so this target is built as C++20.
if(TARGET ssm-hip)
    target_compile_features(ssm-hip PRIVATE hip_std_20)
endif()
