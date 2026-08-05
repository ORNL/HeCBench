.PHONY: all clean run

all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

run: all
	cd src && ./main -nx 128 -ny 128 -nz 128
