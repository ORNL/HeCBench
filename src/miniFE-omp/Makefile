.PHONY: all clean realclean run

LAUNCHER  =

all:
	$(MAKE) -C src EXTRA_CFLAGS="$(EXTRA_CFLAGS)"

clean:
	$(MAKE) -C src clean

realclean:
	$(MAKE) -C src realclean

# Run from src/: miniFE writes its .yaml report into the working directory,
# and only realclean removes it.
run: all
	cd src && $(LAUNCHER) ./main -nx 128 -ny 128 -nz 128
