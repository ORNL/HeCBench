# easyWave-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME easyWave
    MODEL cuda
    SOURCES cOgrd.cpp cOkadaEarthquake.cpp cOkadaFault.cpp cSphere.cpp main.cu okada.cpp utilits.cpp
    CATEGORIES simulation
)
