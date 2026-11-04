# adv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -Ddfloat=double -Ddlong=int
)
