# si-hip was a benchmark that builds via CMake instead of Make.
# This wrapper drives the CMake build out-of-tree under release/ and
# matches the all/clean interface the rest of HeCBench expects.

LAUNCHER  ?=

.PHONY: all clean run

all:
	mkdir -p release
	cd release && cmake $(CURDIR) -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C release

clean:
	rm -rf release

# Encodes the manual run procedure documented in README.md:
#   - generate the input dataset (asc + desc .bin files)
#   - run the main set-intersection benchmark on each
# The dataset depends order-only on `all` (built by CMake as a side effect)
# so Make doesn't try to find a rule for `release/generate_dataset` at parse
# time. The .bin files cache between runs; `make clean` forces regeneration
# along with the rebuild.
DATASET = release/out_asc_uniform_1000_1000_100.bin \
          release/out_desc_uniform_1000_1000_100.bin

$(DATASET): | all
	cd release && ./generate_dataset --iterations 100 --universe 1000 -k 1000

run: $(DATASET)
	cd release && $(LAUNCHER) ./main --input out_asc_uniform_1000_1000_100.bin
	cd release && $(LAUNCHER) ./main --input out_desc_uniform_1000_1000_100.bin
