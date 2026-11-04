# easyWave-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME easyWave
    MODEL sycl
    SOURCES cOgrd.cpp cOkadaEarthquake.cpp cOkadaFault.cpp cSphere.cpp main.cpp okada.cpp utilits.cpp
    CATEGORIES simulation
)
