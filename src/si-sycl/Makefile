# si-sycl was a benchmark that builds via CMake instead of Make.
# This wrapper drives the CMake build out-of-tree under release/ and
# matches the all/clean interface the rest of HeCBench expects.

LAUNCHER  ?=

.PHONY: all clean run

all:
	mkdir -p release
	cd release && cmake $(CURDIR) -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C release

clean:
	rm -rf release

# Encodes the manual run procedure documented in README.md:
#   - generate the input dataset (asc + desc .bin files)
#   - run the main set-intersection benchmark on each
# The dataset is treated as a normal Make prerequisite so it is regenerated
# only when generate_dataset is rebuilt or the .bin files are missing,
# matching the README's two-phase compile/run flow without paying the
# regeneration cost on every rerun.
DATASET = release/out_asc_uniform_1000_1000_100.bin \
          release/out_desc_uniform_1000_1000_100.bin

$(DATASET): | all
	cd release && ./generate_dataset --iterations 100 --universe 1000 -k 1000

run: $(DATASET)
	cd release && $(LAUNCHER) ./main --input out_asc_uniform_1000_1000_100.bin
	cd release && $(LAUNCHER) ./main --input out_desc_uniform_1000_1000_100.bin
