#ifndef _LATTICE_HPP
#define _LATTICE_HPP
// Adapted from lattice.h in MILC version 7

#if defined(USE_OPENCL)  || defined(MILC_COMPLEX)
#  include "su3.h"
#else
#  include "su3.hpp"
#endif

#define EVEN 0x02
#define ODD  0x01

// The lattice is an array of sites.
// alignas is on the struct so std::vector<site> uses this alignment.
// A bare aligned attribute on the typedef name is not part of that type.
typedef struct alignas(16) {
	su3_matrix link[4];  // the fundamental gauge field
	int x,y,z,t;         // coordinates of this site
	int index;           // my index in the array
	char parity;         // is it even or odd?
#if (PRECISION==1)
        int pad[2];
#else
        int pad[10];
#endif
} site;

#endif // _LATTICE_HPP
