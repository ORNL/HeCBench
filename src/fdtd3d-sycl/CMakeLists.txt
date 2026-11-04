# fdtd3d-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fdtd3d
    MODEL sycl
    SOURCES FDTD3dGPU.cpp FDTD3dReference.cpp cmd_arg_reader.cpp main.cpp shrUtils.cpp
    CATEGORIES simulation
)
