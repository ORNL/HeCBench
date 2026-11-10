# particles-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL cuda
    SOURCES bitonicSort.cu main.cu particles.cu
    CATEGORIES simulation
)
