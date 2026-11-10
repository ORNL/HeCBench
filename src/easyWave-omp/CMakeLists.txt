# easyWave-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME easyWave
    MODEL omp
    SOURCES cOgrd.cpp cOkadaEarthquake.cpp cOkadaFault.cpp cSphere.cpp main.cpp okada.cpp utilits.cpp
    CATEGORIES simulation
)
