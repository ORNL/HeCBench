# si-cuda was a benchmark that builds via CMake instead of Make.
# This wrapper drives the CMake build out-of-tree under release/ and
# matches the all/clean interface the rest of HeCBench expects.

ARCH      = sm_60
LAUNCHER  ?=

.PHONY: all clean run

# CUDA Path may need to be set properly first
all:
	mkdir -p release
	cd release && cmake $(CURDIR) -DSM_ARCH=$(subst sm_,,$(ARCH)) -DCMAKE_BUILD_TYPE=Release
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
