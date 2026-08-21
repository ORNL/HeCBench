/*****************************************************************************
  This file is part of the XLQC program.                                      
  Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            

Filename:  basis.cc                                                      
License:   BSD 3-Clause License

This software is provided by the copyright holders and contributors "as is"
and any express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall the copyright holder or contributors be liable
for any direct, indirect, incidental, special, exemplary, or consequential
damages (including, but not limited to, procurement of substitute goods or
services; loss of use, data, or profits; or business interruption) however
caused and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of the use
of this software, even if advised of the possibility of such damage.
 *****************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

#include "typedef.h"
#include "int_lib/cints.h"
#include "int_lib/crys.h"

//=======================================
// allocate memory with failure checking
//=======================================
void* my_malloc(size_t bytes) 
{
  void* ptr = malloc(bytes);
  if(NULL == ptr) 
  {
    fprintf(stderr, "Error: could not allocate memory!\n");
    exit(1);
  } 
  return ptr;
}

void* my_malloc_2(size_t bytes, std::string word)
{
  void* ptr = malloc(bytes);

#ifdef DEBUG
  fprintf(stdout, "size of alloc (%s) = %zu MB\n", word.c_str(), bytes / 1000000);
#endif

  if(ptr == NULL) 
  {
    fprintf(stderr, "Error: could not allocate memory for %s !\n", word.c_str());
    exit(1);
  } 
  return ptr;
}

//=============================
// combination index
//=============================
int ij2intindex(int i, int j)
{
  if (i < j) { return ij2intindex(j, i); }
  return i * (i + 1) / 2 + j;
}

//===================================
// get nuclear charge of an element
//===================================
int get_nuc_chg(char *element)
{
  std::string *periodic = new std::string [89] {"X", 
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", 
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", 
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", 
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", 
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra"};

  std::string elem_name = std::string(element);

  for (int z = 1; z <= 88; ++ z)
  {
    if (0 == elem_name.compare(periodic[z]))
    {
      return z;
    }
  }

  fprintf(stderr, "Error in get_nuc_chg: cannot find element %s!\n", element);
  exit(1);
}

//=============================
// get number of atoms 
//=============================
int get_natoms(void)
{
  FILE *f_geom;
  f_geom = fopen("example/geom.xyz","r");
  if (NULL == f_geom)
  {
    fprintf(stderr, "Error: cannot open file example/geom.xyz!\n");
    exit(1);
  }

  char line[MAX_STR_LEN];
  if (fgets(line, MAX_STR_LEN, f_geom) != NULL)
  {
    int natoms;
    sscanf(line, "%d", &natoms);
    return natoms;
  }
  else
  {
    fprintf(stderr, "Error in the first line of example/geom.xyz!\n");
    exit(1);
  }
}

//================================
// read geometry
//================================
void read_geom(Atom *p_atom)
{
  FILE *f_geom;
  f_geom = fopen("example/geom.xyz","r");
  if (NULL == f_geom)
  {
    fprintf(stderr, "Error: cannot open file example/geom.xyz!\n");
    exit(1);
  }

  int natoms = 0;
  char line[MAX_STR_LEN];
  if (fgets(line, MAX_STR_LEN, f_geom) != NULL)
  {
    sscanf(line, "%d", &natoms);
  }

  if (natoms != p_atom->num)
  {
    fprintf(stderr, "Error: natoms (%d) is not equal to p_atom->num (%d)!\n",
        natoms, p_atom->num);
    exit(1);
  }

  // read comment line
  if (fgets(line, MAX_STR_LEN, f_geom) != NULL) {}

  for (int iatom = 0; iatom < natoms; ++ iatom)
  {
    if (fgets(line, MAX_STR_LEN, f_geom) != NULL)
    {
      // read coordinates in Angstrom
      double rx, ry, rz;
      sscanf(line, "%s%lf%lf%lf", p_atom->name[iatom], &rx, &ry, &rz);

      // save coordinates in Bohr
      p_atom->pos[iatom][0] = rx / BOHR2ANGSTROM;
      p_atom->pos[iatom][1] = ry / BOHR2ANGSTROM;
      p_atom->pos[iatom][2] = rz / BOHR2ANGSTROM;

      p_atom->nuc_chg[iatom] = get_nuc_chg(p_atom->name[iatom]);
    }
  }
}

//================================
// nuclear repulsion energy
//================================
double calc_ene_nucl(Atom *p_atom)
{
  double ene_nucl = 0.0;
  for (int a = 0; a < p_atom->num; ++ a)
  {
    for (int b = a + 1; b < p_atom->num; ++ b)
    {
      double dx = p_atom->pos[a][0] - p_atom->pos[b][0];
      double dy = p_atom->pos[a][1] - p_atom->pos[b][1];
      double dz = p_atom->pos[a][2] - p_atom->pos[b][2];
      double dr = sqrt(dx*dx + dy*dy + dz*dz);
      ene_nucl += (double)p_atom->nuc_chg[a] * 
        (double)p_atom->nuc_chg[b] / dr;
    }
  }
  return ene_nucl;
}

//======================================
// parse basis set
// get number of basis functions
//======================================
void parse_basis(Atom *p_atom, Basis *p_basis, int use_5d)
{
  FILE *f_basis;
  f_basis = fopen("example/basis.dat","r");
  if (NULL == f_basis)
  {
    fprintf(stderr, "Error: cannot open file basis.dat!\n");
    exit(1);
  }

  char line[MAX_STR_LEN], cart_type[5], elem_name[5];
  int  int_num, nprims;
  double  dbl_num;

  // count number of elements
  int ielem = 0;
  // nuclear charge and number of basis functions for each element
  int *elem_nuc_chg = (int *)my_malloc(sizeof(int) * 1);
  int *elem_n_basis = (int *)my_malloc(sizeof(int) * 1);

  // loop over elements
  while (1)
  {
    // reallocate elem_nuc_chg and elem_n_basis 
    // if there are more than one elements
    if (ielem > 0)
    {
      int *tmp;

      tmp = (int *)realloc(elem_nuc_chg, sizeof(int) * (ielem + 1));
      if (tmp != NULL) { elem_nuc_chg = tmp; }

      tmp = (int *)realloc(elem_n_basis, sizeof(int) * (ielem + 1));
      if (tmp != NULL) { elem_n_basis = tmp; }
    }

    // get element name and initialize number of basis functions
    if (fgets(line, MAX_STR_LEN, f_basis) != NULL)
    {
      sscanf(line, "%s%d", elem_name, &int_num);
      elem_nuc_chg[ielem] = get_nuc_chg(elem_name);

      elem_n_basis[ielem] = 0.0;

      if (int_num != 0) 
      { 
        fprintf(stderr, "Error: int_num=%d is not zero!\n", int_num); 
        exit(1);
      }
    }
    else
    {
      break;
    }

    // write basis functions for this element to a separate file
    char elem_file[MAX_STR_LEN];
    sprintf(elem_file, "example/basis_%s.dat", elem_name);

    FILE *f_elem;
    f_elem = fopen(elem_file,"w");
    if (NULL == f_elem)
    {
      fprintf(stderr, "Error: cannot write to file %s!\n", elem_file);
      exit(1);
    }

    fprintf(f_elem, "%s     %d\n", elem_name, int_num);

    // loop over contraction blocks
    while (1)
    {
      if (fgets(line, MAX_STR_LEN, f_basis) != NULL)
      {
        sscanf(line, "%s%d%lf", cart_type, &nprims, &dbl_num);

        if (0 == strcmp(cart_type, "****")) { break; }

        if (dbl_num != 1.00) 
        { 
          fprintf(stderr, "Error: dbl_num=%.2f is not 1.00!\n", dbl_num); 
          exit(1);
        }

        // count number of basis functions for this element
        if      (0 == strcmp(cart_type, "S"))  { elem_n_basis[ielem] += 1; }
        else if (0 == strcmp(cart_type, "SP")) { elem_n_basis[ielem] += 4; }
        else if (0 == strcmp(cart_type, "P"))  { elem_n_basis[ielem] += 3; }
        else if (0 == strcmp(cart_type, "D"))  
        { 
          if (use_5d) { elem_n_basis[ielem] += 5; }
          else        { elem_n_basis[ielem] += 6; }
        }
        else 
        { 
          fprintf(stderr, "Error: Cartesian type %s not supported!\n", cart_type); 
          exit(1); 
        }

        fprintf(f_elem, "%s   %d   %.2f\n", cart_type, nprims, dbl_num);

        for (int iprim = 0; iprim < nprims; ++ iprim)
        {
          if (fgets(line, MAX_STR_LEN, f_basis) != NULL)
          {
            fprintf(f_elem, "%s", line);
          }
        }
      }
    }

    fclose(f_elem);
    fprintf(stdout, "Element %s   Nbasis %d\n", elem_name, elem_n_basis[ielem]);

    // update counter for elements
    ++ ielem;
  }

  // total number of elements
  int nelems = ielem;

  fclose(f_basis);


  // initialize number of basis functions for the whole system
  p_basis->num = 0;

  // write basis functions for each atom into basis_all.dat
  FILE *f_basis_all;
  f_basis_all = fopen("example/basis_all.dat", "w");
  if (NULL == f_basis_all)
  {
    fprintf(stderr, "Error: cannot write to file basis_all.dat!\n");
    exit(1);
  }

  for (int iatom = 0; iatom < p_atom->num; ++ iatom)
  {
    // find the correct element for this atom
    // and count number of basis functions
    int found = 0;
    for (int ielem = 0; ielem < nelems; ++ ielem)
    {
      if (p_atom->nuc_chg[iatom] == elem_nuc_chg[ielem])
      {
        p_basis->num += elem_n_basis[ielem];
        found = 1;
        break;
      }
    }

    if (0 == found)
    {
      fprintf(stderr, "Error: no basis function for atom %d (%s)!\n", iatom+1, p_atom->name[iatom]);
      exit(1);
    }

    // read basis set for this atom
    // and write to basis_all.dat
    char atom_file[MAX_STR_LEN];
    sprintf(atom_file, "example/basis_%s.dat", p_atom->name[iatom]);

    FILE *f_atom;
    f_atom = fopen(atom_file,"r");
    if (NULL == f_atom)
    {
      fprintf(stderr, "Error: cannot open file %s!\n", atom_file);
      exit(1);
    }

    while (fgets(line, MAX_STR_LEN, f_atom) != NULL)
    {
      fprintf(f_basis_all, "%s", line);
    }
    fclose(f_atom);

    fprintf(f_basis_all, "****\n");
  }

  fclose(f_basis_all);


  free(elem_nuc_chg);
  free(elem_n_basis);
}

//==================================================
// read the full basis set created by parse_basis
//==================================================
void read_basis(Atom *p_atom, Basis *p_basis, int use_5d)
{
  FILE *f_basis_all;
  f_basis_all = fopen("example/basis_all.dat","r");
  if (NULL == f_basis_all)
  {
    fprintf(stderr, "Error: cannot open file basis_all.dat!\n");
    exit(1);
  }

  char line[MAX_STR_LEN], cart_type[5];

  // counter for basis functions and atoms
  int ibasis = 0;
  int iatom = 0;

  // Cartesian l,m,n
  int s_lmn[N_S * CART_DIM]   = {0,0,0};
  int sp_lmn[N_SP * CART_DIM] = {0,0,0,  1,0,0,  0,1,0,  0,0,1};
  int p_lmn[N_P * CART_DIM]   = {1,0,0,  0,1,0,  0,0,1};
  // XY,  YZ,  ZZ,  XZ,  XX  (note: ZZ and XX will be replaced by spherical harmonics)
  // D-2, D-1, D 0, D+1, D+2
  int d_lmn[N_D * CART_DIM]   = {1,1,0,  0,1,1,  0,0,2,  1,0,1,  2,0,0};
  // XX, XY, XZ, YY, YZ, ZZ
  int d_lmn_cart[N_D_CART * CART_DIM] = {2,0,0,  1,1,0,  1,0,1,  0,2,0,  0,1,1,  0,0,2};

  // loop over elements
  while (1)
  {
    // first line; H   0
    if (NULL == fgets(line, MAX_STR_LEN, f_basis_all))
    {
      break;
    }

    // loop over blocks
    while (1)
    {
      if (fgets(line, MAX_STR_LEN, f_basis_all) != NULL)
      {
        double dbl_num;
        int    nprims;
        sscanf(line, "%s%d%lf", cart_type, &nprims, &dbl_num);

        if (0 == strcmp(cart_type, "****")) { break; }

        int N = 0;
        int num_basis = 0;
        int *ptr_lmn = NULL;
        if      (0 == strcmp(cart_type, "S"))  { N = N_S;  ptr_lmn = &s_lmn[0];  num_basis = 1; }
        else if (0 == strcmp(cart_type, "SP")) { N = N_SP; ptr_lmn = &sp_lmn[0]; num_basis = 4; }
        else if (0 == strcmp(cart_type, "P"))  { N = N_P;  ptr_lmn = &p_lmn[0];  num_basis = 3; }
        else if (0 == strcmp(cart_type, "D"))  
        { 
          if (use_5d) { N = N_D;  ptr_lmn = &d_lmn[0];  num_basis = 5; }
          else { N = N_D_CART;  ptr_lmn = &d_lmn_cart[0];  num_basis = 6; }
        }

        for (int iprim = 0; iprim < nprims; ++ iprim)
        {
          if (fgets(line, MAX_STR_LEN, f_basis_all) != NULL)
          {
            double expon_1, coef_1, coef_2;
            coef_2 = 0.0;
            sscanf(line, "%lf%lf%lf", &expon_1, &coef_1, &coef_2);


            // Note: rules for converting Cartesian d to spherical harmonic d
            // Cartesian d function
            // XY, YZ, ZZ, XZ, (YY,) XX
            // Spheircal harmonic d function
            // D-2 = XY
            // D-1 = YZ
            // D 0 = -0.5*XX-0.5*YY+ZZ
            // D+1 = XZ
            // D+2 = sqrt(3)/2*(XX-YY)
            // Convert: ZZ->D0, XX->D+2


            for (int ii = 0; ii < N; ++ ii)
            {
              // allocate memories at the beginning
              if (0 == iprim)
              {
                // assign number of primitive basis functions
                p_basis->nprims[ibasis + ii] = nprims;

                // take care of 5d functions
                if (use_5d && N_D == N)
                {
                  // D 0 = -0.5*XX-0.5*YY+ZZ
                  if      (2 == ii) { p_basis->nprims[ibasis + ii] = nprims * 3; }
                  // D+2 = sqrt(3)/2*(XX-YY)
                  else if (4 == ii) { p_basis->nprims[ibasis + ii] = nprims * 2; }
                }

                // allocate memories
                size_t bytes_dbl = sizeof(double) * p_basis->nprims[ibasis + ii];
                size_t bytes_int = sizeof(int)    * p_basis->nprims[ibasis + ii];

                p_basis->expon[ibasis + ii] = (double *)my_malloc(bytes_dbl);
                p_basis->coef[ibasis + ii]  = (double *)my_malloc(bytes_dbl);

                p_basis->lx[ibasis + ii] = (int *)my_malloc(bytes_int);
                p_basis->ly[ibasis + ii] = (int *)my_malloc(bytes_int);
                p_basis->lz[ibasis + ii] = (int *)my_malloc(bytes_int);

                // assign centers of basis functions
                p_basis->xbas[ibasis + ii] = p_atom->pos[iatom][0];
                p_basis->ybas[ibasis + ii] = p_atom->pos[iatom][1];
                p_basis->zbas[ibasis + ii] = p_atom->pos[iatom][2];
              }

              // after memory allocation, assign values for
              // expon, coef, and lx,ly,lz

              // assign exponents
              p_basis->expon[ibasis + ii][iprim] = expon_1;

              // assign coefficients; take care of sp functions
              if (0 == strcmp(cart_type, "SP") && ii > 0) {
                p_basis->coef[ibasis + ii][iprim] = coef_2;
              } else {
                p_basis->coef[ibasis + ii][iprim] = coef_1;
              }

              // assign lx,ly,lz
              p_basis->lx[ibasis + ii][iprim] = ptr_lmn[ii * CART_DIM + 0];
              p_basis->ly[ibasis + ii][iprim] = ptr_lmn[ii * CART_DIM + 1];
              p_basis->lz[ibasis + ii][iprim] = ptr_lmn[ii * CART_DIM + 2];

              // take care of 5d functions
              if (use_5d && N_D == N)
              {
                // D 0 = -0.5*XX-0.5*YY+ZZ
                if (2 == ii)
                {
                  p_basis->expon[ibasis + ii][iprim] = expon_1;
                  p_basis->coef[ibasis + ii][iprim]  = -0.5 * coef_1;

                  p_basis->lx[ibasis + ii][iprim] = 2;
                  p_basis->ly[ibasis + ii][iprim] = 0;
                  p_basis->lz[ibasis + ii][iprim] = 0;

                  p_basis->expon[ibasis + ii][iprim + nprims] = expon_1;
                  p_basis->coef[ibasis + ii][iprim + nprims]  = -0.5 * coef_1;

                  p_basis->lx[ibasis + ii][iprim + nprims] = 0;
                  p_basis->ly[ibasis + ii][iprim + nprims] = 2;
                  p_basis->lz[ibasis + ii][iprim + nprims] = 0;

                  p_basis->expon[ibasis + ii][iprim + nprims*2] = expon_1;
                  p_basis->coef[ibasis + ii][iprim + nprims*2]  = coef_1;

                  p_basis->lx[ibasis + ii][iprim + nprims*2] = 0;
                  p_basis->ly[ibasis + ii][iprim + nprims*2] = 0;
                  p_basis->lz[ibasis + ii][iprim + nprims*2] = 2;
                }
                // D+2 = sqrt(3)/2*(XX-YY)
                else if (4 == ii)
                {
                  double prefac = sqrt(3.0) / 2.0;

                  p_basis->expon[ibasis + ii][iprim] = expon_1;
                  p_basis->coef[ibasis + ii][iprim]  = prefac * coef_1;

                  p_basis->lx[ibasis + ii][iprim] = 2;
                  p_basis->ly[ibasis + ii][iprim] = 0;
                  p_basis->lz[ibasis + ii][iprim] = 0;

                  p_basis->expon[ibasis + ii][iprim + nprims] = expon_1;
                  p_basis->coef[ibasis + ii][iprim + nprims]  = -prefac * coef_1;

                  p_basis->lx[ibasis + ii][iprim + nprims] = 0;
                  p_basis->ly[ibasis + ii][iprim + nprims] = 2;
                  p_basis->lz[ibasis + ii][iprim + nprims] = 0;
                }
              }
            }
          }
        }

        ibasis += num_basis;
      }
    }

    ++ iatom;
  }

  fclose(f_basis_all);

  // check number of atoms and basis functions
  if (iatom != p_atom->num)
  {
    fprintf(stderr, "Error: iatom=%d, p_atom->num=%d\n", iatom, p_atom->num);
  }
  if (ibasis != p_basis->num)
  {
    fprintf(stderr, "Error: ibasis=%d, p_basis->num=%d\n", ibasis, p_basis->num);
  }

  // calculate normalization factors
  for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
  {
    p_basis->norm[ibasis] = (double *)my_malloc(sizeof(double) * p_basis->nprims[ibasis]);

    int iprim;
    for (iprim = 0; iprim < p_basis->nprims[ibasis]; ++ iprim)
    {
      p_basis->norm[ibasis][iprim] = 
        norm_factor(p_basis->expon[ibasis][iprim], p_basis->lx[ibasis][iprim], 
            p_basis->ly[ibasis][iprim], p_basis->lz[ibasis][iprim]);
    }
  }
}

//================================
// print the basis set
//================================
void print_basis(Basis *p_basis)
{
  int ibasis;
  for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
  {
    int iprim;
    for (iprim = 0; iprim < p_basis->nprims[ibasis]; ++ iprim)
    {
      /*
         printf("%16.8f%16.8f%16.8f ", 
         p_basis->xbas[ibasis], p_basis->ybas[ibasis], p_basis->ybas[ibasis]);
         */
      printf("%16.8f%16.8f %5d%5d%5d   ibf_%-3d  ipm_%-3d  npm=%d\n", 
          p_basis->expon[ibasis][iprim], p_basis->coef[ibasis][iprim],
          p_basis->lx[ibasis][iprim], p_basis->ly[ibasis][iprim], p_basis->lz[ibasis][iprim],
          ibasis, iprim, p_basis->nprims[ibasis]);
    }
  }
}

//================================
// overlap integral
//================================
double calc_int_overlap(Basis *p_basis, int a, int b)
{
  double s;

  s = contr_overlap(
      p_basis->nprims[a], p_basis->expon[a], p_basis->coef[a], p_basis->norm[a], 
      p_basis->xbas[a], p_basis->ybas[a], p_basis->zbas[a], 
      p_basis->lx[a], p_basis->ly[a], p_basis->lz[a],

      p_basis->nprims[b], p_basis->expon[b], p_basis->coef[b], p_basis->norm[b], 
      p_basis->xbas[b], p_basis->ybas[b], p_basis->zbas[b], 
      p_basis->lx[b], p_basis->ly[b], p_basis->lz[b]);

  return s;
}

//================================
// kinetic energy integral
//================================
double calc_int_kinetic(Basis *p_basis, int a, int b)
{
  double t;

  t = contr_kinetic(
      p_basis->nprims[a], p_basis->expon[a], p_basis->coef[a], p_basis->norm[a], 
      p_basis->xbas[a], p_basis->ybas[a], p_basis->zbas[a], 
      p_basis->lx[a], p_basis->ly[a], p_basis->lz[a],

      p_basis->nprims[b], p_basis->expon[b], p_basis->coef[b], p_basis->norm[b], 
      p_basis->xbas[b], p_basis->ybas[b], p_basis->zbas[b], 
      p_basis->lx[b], p_basis->ly[b], p_basis->lz[b]);

  return t;
}

//=======================================
// nulcear-electron attraction integral
//=======================================
double calc_int_nuc_attr(Basis *p_basis, int a, int b, Atom *p_atom)
{
  double v;

  v = contr_nuc_attr(
      p_basis->nprims[a], p_basis->expon[a], p_basis->coef[a], p_basis->norm[a], 
      p_basis->xbas[a], p_basis->ybas[a], p_basis->zbas[a], 
      p_basis->lx[a], p_basis->ly[a], p_basis->lz[a],

      p_basis->nprims[b], p_basis->expon[b], p_basis->coef[b], p_basis->norm[b], 
      p_basis->xbas[b], p_basis->ybas[b], p_basis->zbas[b], 
      p_basis->lx[b], p_basis->ly[b], p_basis->lz[b],

      p_atom->num, p_atom->nuc_chg, p_atom->pos);

  return v;
}

//===========================
// two-electron integrals
//===========================
double calc_int_eri_rys(Basis *p_basis, int a, int b, int c, int d)
{
  double eri;

  eri = rys_contr_coulomb(
      p_basis->nprims[a], p_basis->expon[a], p_basis->coef[a], p_basis->norm[a], 
      p_basis->xbas[a], p_basis->ybas[a], p_basis->zbas[a], 
      p_basis->lx[a], p_basis->ly[a], p_basis->lz[a],

      p_basis->nprims[b], p_basis->expon[b], p_basis->coef[b], p_basis->norm[b], 
      p_basis->xbas[b], p_basis->ybas[b], p_basis->zbas[b], 
      p_basis->lx[b], p_basis->ly[b], p_basis->lz[b],

      p_basis->nprims[c], p_basis->expon[c], p_basis->coef[c], p_basis->norm[c], 
      p_basis->xbas[c], p_basis->ybas[c], p_basis->zbas[c], 
      p_basis->lx[c], p_basis->ly[c], p_basis->lz[c],

      p_basis->nprims[d], p_basis->expon[d], p_basis->coef[d], p_basis->norm[d],
      p_basis->xbas[d], p_basis->ybas[d], p_basis->zbas[d], 
      p_basis->lx[d], p_basis->ly[d], p_basis->lz[d]);

  return eri;
}
