experimental quantum chemistry code.
Illustrated with a Hartree-Fock SCF calculation of H2O.
Now use GPU for two-electron integral calculations.

Integral subroutines taken from:
"PyQuante: Python Quantum Chemistry"
http://pyquante.sourceforge.net/

Refereces for Quantum Chemistry on GPU:
a) ERI evaluation. http://pubs.acs.org/doi/abs/10.1021/ct700268q
b) Direct SCF. http://pubs.acs.org/doi/abs/10.1021/ct800526s

Dependencies
------------
The Hartree-Fock SCF driver needs a dense linear algebra library for the
symmetric eigensolver, matrix products and the DIIS linear solve. These are
provided by Eigen (header-only) through the thin `gsl_compat.h` wrapper in
`xlqc-cuda/`. The Makefile discovers Eigen with `pkg-config --cflags eigen3`,
so an Eigen installed by the system package manager (`libeigen3-dev` on Debian
and Ubuntu, `eigen3-devel` on Fedora) needs no configuration. For an Eigen in a
non-standard prefix, either add it to `PKG_CONFIG_PATH` or override the include
path directly, for example `make EIGEN_INC=-I/opt/eigen-3.4.0/include/eigen3`.
