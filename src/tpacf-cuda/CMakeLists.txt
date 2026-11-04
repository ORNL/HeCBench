# tpacf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tpacf
    MODEL cuda
    SOURCES main.cu args.cu compute.cu
    CATEGORIES simulation
)
