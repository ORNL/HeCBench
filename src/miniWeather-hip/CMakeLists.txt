# miniWeather-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME miniWeather
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    COMPILE_OPTIONS -DNX=400 -DNZ=200 -DSIM_TIME=600 -DDATA_SPEC=DATA_SPEC_THERMAL
)
