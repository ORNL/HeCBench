/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <iostream>
#include <omp.h>

#include "gsl_compat.h"

#include "int_lib/cints.h"
#include "int_lib/crys.h"

#include "typedef.h"
#include "basis.h"
#include "scf.h"

#include "rys.h"
#include "cuda_rys_sp.cpp"
#include "cuda_rys_dp.cpp"

int main(int argc, char* argv[])
{
    // use spherical harmonic d function?
    int use_5d = 1;
    // use double precision?
    int use_dp = 1;
    // status is set on failure of result check
    int status = 0;

    if (argc > 1) {
        for (int i = 1; i < argc; ++ i) {
            if (0 == strcmp(argv[i],"sp")) { use_dp = 0; }
            if (0 == strcmp(argv[i],"dp")) { use_dp = 1; }
            if (0 == strcmp(argv[i],"6d")) { use_5d = 0; }
            if (0 == strcmp(argv[i],"5d")) { use_5d = 1; }
        }
    }

    // initialize timer
    double time_in_usec, time_total, time_mat_J, time_mat_K;

    auto start = std::chrono::steady_clock::now();
    std::string time_txt ("");
    time_total = 0.0;
    time_mat_J = 0.0;
    time_mat_K = 0.0;

    Atom   *p_atom   = (Atom *)my_malloc(sizeof(Atom) * 1);
    Basis  *p_basis  = (Basis *)my_malloc(sizeof(Basis) * 1);


    //====== parse geom.dat ========

    // get number of atoms
    p_atom->num = get_natoms();
    fprintf(stdout, "Natoms = %d\n", p_atom->num);

    // atomic coordinates and atom name
    p_atom->pos = (double **)my_malloc(sizeof(double *) * p_atom->num);
    p_atom->name = (char **)my_malloc(sizeof(char *) * p_atom->num);

    for (int iatom = 0; iatom < p_atom->num; ++ iatom)
    {
        p_atom->pos[iatom] = (double *)my_malloc(sizeof(double) * CART_DIM);
        p_atom->name[iatom] = (char *)my_malloc(sizeof(char) * 5);
    }

    // nuclear charge
    p_atom->nuc_chg = (int *)my_malloc(sizeof(int) * p_atom->num);

    // read atomic positions, nuclear charge and atom name
    read_geom(p_atom);

    fprintf(stdout, "Coordinates in atomic unit:\n");
    for (int iatom = 0; iatom < p_atom->num; ++ iatom)
    {
        fprintf(stdout, "%s (%.1f)  %.10f  %.10f  %.10f\n", 
                p_atom->name[iatom], (double)p_atom->nuc_chg[iatom],
                p_atom->pos[iatom][0], p_atom->pos[iatom][1], p_atom->pos[iatom][2]);
    }
    
    // nuclear repulsion energy
    double ene_nucl = calc_ene_nucl(p_atom);
    fprintf(stdout, "Nuclear repulsion = %-20.10f\n", ene_nucl);


    //====== parse basis.dat ========

    // parse basis functions
    parse_basis(p_atom, p_basis, use_5d);

    fprintf(stdout, "System Nbasis = %d\n", p_basis->num);

    // basis function exponents, coefficients, and normalization factors
    p_basis->expon = (double **)my_malloc(sizeof(double *) * p_basis->num);
    p_basis->coef  = (double **)my_malloc(sizeof(double *) * p_basis->num);
    p_basis->norm  = (double **)my_malloc(sizeof(double *) * p_basis->num);

    // number of primitive functions in each contracted funciton
    p_basis->nprims = (int *)my_malloc(sizeof(int) * p_basis->num);

    // Cartesian coordinates and l,m,n numbers
    p_basis->xbas  = (double *)my_malloc(sizeof(double) * p_basis->num);
    p_basis->ybas  = (double *)my_malloc(sizeof(double) * p_basis->num);
    p_basis->zbas  = (double *)my_malloc(sizeof(double) * p_basis->num);

    p_basis->lx = (int **)my_malloc(sizeof(int *) * p_basis->num);
    p_basis->ly = (int **)my_malloc(sizeof(int *) * p_basis->num);
    p_basis->lz = (int **)my_malloc(sizeof(int *) * p_basis->num);

    // read basis set (also calculate normalization factors)
    read_basis(p_atom, p_basis, use_5d);

#ifdef DEBUG
    print_basis(p_basis);
#endif

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_in_usec = time * 1e-3f;
    time_txt += "Time_Basis    = " + std::to_string(time_in_usec) + " usec\n";
    time_total += time_in_usec;


    //====== one-electron integrals ========

    start = std::chrono::steady_clock::now();

    // overlap, kinetic energy and nuclear attraction integral
    gsl_matrix *S = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_matrix *T = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_matrix *V = gsl_matrix_alloc(p_basis->num, p_basis->num);

    for (int a = 0; a < p_basis->num; ++ a)
    {
        for (int b = 0; b <= a; ++ b)
        {
            // overlap
            double s = calc_int_overlap(p_basis, a, b);

            // kinetic energy
            double t = calc_int_kinetic(p_basis, a, b);

            // nuclear repulsion
            double v = calc_int_nuc_attr(p_basis, a, b, p_atom);

            // save one-electron integrals in matrices
            gsl_matrix_set(S, a, b, s);
            gsl_matrix_set(T, a, b, t);
            gsl_matrix_set(V, a, b, v);
            if (a != b)
            {
                gsl_matrix_set(S, b, a, s);
                gsl_matrix_set(T, b, a, t);
                gsl_matrix_set(V, b, a, v);
            }
        }
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_in_usec = time * 1e-3f;
    time_txt += "Time_1e_Ints  = " + std::to_string(time_in_usec) + " usec\n";
    time_total += time_in_usec;

    //====== allocate memory for arrays on host ========

    start = std::chrono::steady_clock::now();

    // number of primitive basis functions (pbf)
    int n_pbf = 0;
    for (int a = 0; a < p_basis->num; ++ a) {
        n_pbf += p_basis->nprims[a];
    }

    // number of unique pbf pairs
    int n_pbf_combi = n_pbf * (n_pbf + 1) / 2;

    size_t n_PBF_bytes  = sizeof(double) * n_pbf;
    size_t n_PBF_bytes_int = sizeof(int) * n_pbf;
    size_t n_PI_bytes   = sizeof(double) * n_pbf_combi;

    // pbf_xlec contains information of each pbf: x,y,z, lx,ly,lz, expon, coef*norm
    // pbf_to_cbf returns index of contracted basis function (cbf) of each pbf
    double *h_pbf_xlec = (double *)my_malloc(n_PBF_bytes * 8);
    int  *h_pbf_to_cbf = (int *)my_malloc(n_PBF_bytes_int);

    // mat_J_PI and mat_K_PI are primitive J and K matrices
    double *h_mat_J_PI = (double *)my_malloc(n_PI_bytes);
    double *h_mat_K_PI = (double *)my_malloc(n_PI_bytes);

    // counter for pbf_xlec; i_pbf for pbf_to_cbf
    int counter = 0;
    int i_pbf = 0;
    for (int a = 0; a < p_basis->num; ++ a) 
    {
        for (int i = 0; i < p_basis->nprims[a]; ++ i)
        {
            h_pbf_to_cbf[i_pbf] = a; ++ i_pbf;

            h_pbf_xlec[counter] = p_basis->xbas[a]; ++ counter;
            h_pbf_xlec[counter] = p_basis->ybas[a]; ++ counter;
            h_pbf_xlec[counter] = p_basis->zbas[a]; ++ counter;

            h_pbf_xlec[counter] = (double)p_basis->lx[a][i]; ++ counter;
            h_pbf_xlec[counter] = (double)p_basis->ly[a][i]; ++ counter;
            h_pbf_xlec[counter] = (double)p_basis->lz[a][i]; ++ counter;

            h_pbf_xlec[counter] = p_basis->expon[a][i]; ++ counter;
            h_pbf_xlec[counter] = p_basis->coef[a][i] * p_basis->norm[a][i]; ++ counter;
            // note that 'norm' is absorbed into 'coef'
        }
    }
    assert(counter == n_pbf * 8);


    // number of unique pairs of contracted basis functions
    int n_combi = p_basis->num * (p_basis->num + 1) / 2;

    size_t n_CI_bytes = sizeof(double) * n_combi;

    // D: density matrix
    // J: Coulomb matrix
    // K: exchange matrix
    // Q: Schwartz pre-screening matrix
    double *h_mat_D = (double *)my_malloc(n_CI_bytes);
    double *h_mat_J = (double *)my_malloc(n_CI_bytes);
    double *h_mat_K = (double *)my_malloc(n_CI_bytes);
    double *h_mat_Q = (double *)my_malloc(n_CI_bytes);


    // memory usage on device
    size_t mem_on_dev = n_PBF_bytes*8 + n_PBF_bytes_int + n_PI_bytes*2 + n_CI_bytes*2;
    fprintf(stdout, "Mem_on_Device = ");
    if   (mem_on_dev > 1000000000) { fprintf(stdout, "%zu GB\n", mem_on_dev / 1000000000); }
    else if (mem_on_dev > 1000000) { fprintf(stdout, "%zu MB\n", mem_on_dev / 1000000); }
    else if (mem_on_dev > 1000)    { fprintf(stdout, "%zu KB\n", mem_on_dev / 1000); }
    else                           { fprintf(stdout, "%zu B\n",  mem_on_dev); }


    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_in_usec = time * 1e-3f;
    time_txt += "Time_2e_Prep  = " + std::to_string(time_in_usec) + " usec\n";
    time_total += time_in_usec;

    //====== start SCF calculation ========

    start = std::chrono::steady_clock::now();

    // NOTE: assume zero charge and closed-shell electronics structure
    int n_elec = 0;
    for (int iatom = 0; iatom < p_atom->num; ++ iatom) {
        n_elec += p_atom->nuc_chg[iatom];
    }

    if (n_elec % 2 != 0) {
        fprintf(stderr, "Error: Number of electrons (%d) is not even!\n", n_elec);
    }

    int n_occ = n_elec / 2;


    // get core Hamiltonian
    gsl_matrix *H_core = gsl_matrix_alloc(p_basis->num, p_basis->num);
    sum_H_core(p_basis->num, H_core, T, V);

    // get S^-1/2
    gsl_matrix *S_invsqrt = gsl_matrix_alloc(p_basis->num, p_basis->num);
    diag_overlap(p_basis->num, S, S_invsqrt);

#ifdef DEBUG
    printf("S:\n"); my_print_matrix(S);
    printf("T:\n"); my_print_matrix(T);
    printf("V:\n"); my_print_matrix(V);
    printf("H_core:\n"); my_print_matrix(H_core);
    printf("S^-1/2:\n"); my_print_matrix(S_invsqrt);
#endif

    // matrices, vector and variables to be used in SCF
    gsl_matrix *D      = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_matrix *D_prev = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_matrix *D_diff = gsl_matrix_alloc(p_basis->num, p_basis->num);

    gsl_matrix *Fock      = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_matrix *Fock_prev = gsl_matrix_alloc(p_basis->num, p_basis->num);

    gsl_matrix *Coef   = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_vector *emo    = gsl_vector_alloc(p_basis->num);

    double ene_elec, ene_total, ene_prev;

    // Coulomb(J) and exchange(K) matrices
    gsl_matrix *J = gsl_matrix_alloc(p_basis->num, p_basis->num);
    gsl_matrix *K = gsl_matrix_alloc(p_basis->num, p_basis->num);

    // initialize density matrix
    gsl_matrix_set_zero(D_prev);
    gsl_matrix_set_zero(D);
    ene_prev = 0.0;


    // Generalized Wolfsberg-Helmholtz initial guess
    init_guess_GWH(p_basis, H_core, S, Fock);
    Fock_to_Coef(p_basis->num, Fock, S_invsqrt, Coef, emo);
    Coef_to_Dens(p_basis->num, n_occ, Coef, D_prev);

    gsl_matrix_memcpy(D_diff, D_prev);
    gsl_matrix_memcpy(Fock_prev, Fock);


    // DIIS error and Fock matrices
    double ***diis_err  = (double ***)my_malloc(sizeof(double **) * MAX_DIIS_DIM);
    double ***diis_Fock = (double ***)my_malloc(sizeof(double **) * MAX_DIIS_DIM);
    int idiis, ibasis;
    for (idiis = 0; idiis < MAX_DIIS_DIM; ++ idiis)
    {
        diis_err[idiis]  = (double **)my_malloc(sizeof(double *) * p_basis->num);
        diis_Fock[idiis] = (double **)my_malloc(sizeof(double *) * p_basis->num);
        for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
        {
            diis_err[idiis][ibasis]  = (double *)my_malloc(sizeof(double) * p_basis->num);
            diis_Fock[idiis][ibasis] = (double *)my_malloc(sizeof(double) * p_basis->num);
        }
    }

    // DIIS index and dimension
    int diis_index = 0;
    int diis_dim = 0;
    double delta_DIIS;

    fprintf(stdout, "%5s %20s %20s %20s %20s\n",
            "Iter", "E_total", "delta_E", "rms_D", "delta_DIIS");


    // mat_Q: sqrt(ab|ab) for prescreening of two-electron integrals
    for (int a = 0; a < p_basis->num; ++ a) {
        for (int b = 0; b <= a; ++ b) {
            h_mat_Q[ij2intindex(a,b)] = calc_int_eri_rys(p_basis, a, b, a, b);
        }
    }

  #pragma omp target data map (to: h_pbf_xlec[0:n_pbf*8], \
                                   h_pbf_to_cbf[0:n_pbf],\
                                   h_mat_Q[0:n_combi]) \
                          map (alloc: h_mat_J_PI[0:n_pbf_combi], \
                                      h_mat_K_PI[0:n_pbf_combi],\
                                      h_mat_D[0:n_combi])
  {
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_in_usec = time * 1e-3f;
    time_txt += "Time_SCF_Init = " + std::to_string(time_in_usec) + " usec\n";
    time_total += time_in_usec;

    // start SCF iterations

    start = std::chrono::steady_clock::now();

    int iter = 0;
    while (1)
    {
        /*------------------------------------*
         * SCF procedure:
         * Form new Fock matrix
         * F' = S^-1/2 * F * S^-1/2
         * diagonalize F' matrix to get C'
         * C = S^-1/2 * C'
         * compute new density matrix 
         *------------------------------------*/

        // when iter > 0, use incremental Fock matrix formation and DIIS
        int use_incr_fock = iter;
        int use_diis = 0;
        if (p_basis->num > 5) { use_diis = iter; } // sometimes DIIS does not work well


        // copy density matrix to device
        for (int a = 0; a < p_basis->num; ++ a) {
            for (int b = 0; b <= a; ++ b) {
                if (use_incr_fock) { h_mat_D[ij2intindex(a,b)] = gsl_matrix_get(D_diff,a,b); }
                else               { h_mat_D[ij2intindex(a,b)] = gsl_matrix_get(D_prev,a,b); }
            }
        }

        #pragma omp target update to (h_mat_D[0:n_combi])

        // timer for J and K matrices
        auto kstart = std::chrono::steady_clock::now();

        // use 1T1PI for J and K matrices
        if (use_dp) {
            cuda_mat_J_PI_dp
                (h_pbf_xlec, h_pbf_to_cbf, n_pbf, h_mat_D, h_mat_J_PI, h_mat_Q);
        } else {
            cuda_mat_J_PI
                (h_pbf_xlec, h_pbf_to_cbf, n_pbf, h_mat_D, h_mat_J_PI, h_mat_Q);
        }

        auto kend = std::chrono::steady_clock::now();
        auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
        time_in_usec = ktime * 1e-3f;
        time_mat_J += time_in_usec;

        #pragma omp target update from (h_mat_J_PI[0:n_pbf_combi])

        kstart = std::chrono::steady_clock::now();

        if (use_dp) {
            cuda_mat_K_PI_dp
                (h_pbf_xlec, h_pbf_to_cbf, n_pbf, h_mat_D, h_mat_K_PI, h_mat_Q);
        } else {
            cuda_mat_K_PI
                (h_pbf_xlec, h_pbf_to_cbf, n_pbf, h_mat_D, h_mat_K_PI, h_mat_Q);
        }

        kend = std::chrono::steady_clock::now();
        ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
        time_in_usec = ktime * 1e-3f;
        time_mat_K += time_in_usec;

        #pragma omp target update from (h_mat_K_PI[0:n_pbf_combi])

        // sum up primitive J and K matrices to contracted ones
        for (int a = 0; a < p_basis->num; ++ a) {
            for (int b = 0; b <= a; ++ b) {
                int ab = ij2intindex(a,b);
                h_mat_J[ab] = 0.0;
                h_mat_K[ab] = 0.0;
            }
        }

        for (int i = 0; i < n_pbf; ++ i)
        {
            int a = h_pbf_to_cbf[i];
            for (int j = 0; j < n_pbf; ++ j)
            {
                int b = h_pbf_to_cbf[j];
                if (a < b) { continue; }

                int ab = ij2intindex(a,b);
                int ij = ij2intindex(i,j);
                h_mat_J[ab] += h_mat_J_PI[ij];
                h_mat_K[ab] += h_mat_K_PI[ij];
            }
        }


        // use J and K matrix from GPU
        for (int a = 0; a < p_basis->num; ++ a) {
            for (int b = 0; b < p_basis->num; ++ b) {
                int ab = ij2intindex(a,b);
                gsl_matrix_set(J,a,b,h_mat_J[ab]);
                gsl_matrix_set(K,a,b,h_mat_K[ab]);
            }
        }


#ifdef DEBUG
        printf("J:\n"); my_print_matrix(J);
        printf("K:\n"); my_print_matrix(K);
#endif

        if (use_incr_fock) { form_Fock(p_basis->num, Fock_prev, J, K, Fock); }
        else               { form_Fock(p_basis->num, H_core, J, K, Fock); }

        // save Fock_prev at this point, so as not to mix with DIIS
        gsl_matrix_memcpy(Fock_prev, Fock);


        // DIIS
        if (use_diis)
        {
            update_Fock_DIIS(&diis_dim, &diis_index, &delta_DIIS, 
                Fock, D_prev, S, p_basis, diis_err, diis_Fock);
        }

        // update density matrix and energies
        Fock_to_Coef(p_basis->num, Fock, S_invsqrt, Coef, emo);
        Coef_to_Dens(p_basis->num, n_occ, Coef, D);

        ene_elec = get_elec_ene(p_basis->num, D, H_core, Fock);
        ene_total = ene_nucl + ene_elec;

#ifdef DEBUG
        printf("F:\n"); my_print_matrix(Fock);
        printf("C:\n"); my_print_matrix(Coef);
        printf("P:\n"); my_print_matrix(D);
#endif

        // check convergence
        double delta_E = ene_total - ene_prev;

        double rms_D = 0.0;
        double dd_max = 0.0;
        int mu, nu;
        for (mu = 0; mu < p_basis->num; ++ mu)
        {
            for (nu = 0; nu < p_basis->num; ++ nu)
            {
                double dd = gsl_matrix_get(D, mu, nu) - 
                            gsl_matrix_get(D_prev, mu, nu);

                gsl_matrix_set(D_diff, mu, nu, dd);

                rms_D += dd * dd;
                if (fabs(dd) > dd_max) { dd_max = fabs(dd); }
            }
        }
        rms_D = sqrt(rms_D) / p_basis->num;

        fprintf(stdout, "%5d %20.10f", iter, ene_total);
        if (iter > 0) { fprintf(stdout, " %20.10f %20.10f", delta_E, rms_D); }
        if (use_diis && iter > 1) { fprintf(stdout, " %20.10f", delta_DIIS); }
        fprintf(stdout, "\n");

        // convergence criteria
        if (fabs(delta_E/ene_total) < 1.0e-9 && rms_D < 1.0e-7 && dd_max < 1.0e-6) { break; }

        // update energy and density matrix for the next iteration
        ene_prev = ene_total;
        gsl_matrix_memcpy(D_prev, D);

        // count iterations
        ++ iter;
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_in_usec = time * 1e-3f;
    time_txt += "Time_SCF_Conv = " + std::to_string(time_in_usec) + " usec\n";
    time_total += time_in_usec;

    // Validate the RHF/STO-3G energy of H2O (geometry in example/geom.xyz).
    // Reference: Szabo & Ostlund, "Modern Quantum Chemistry", Table 3.13 lists the
    // HF/STO-3G total energy of H2O as -74.963 Eh at the standard geometry of
    // Table 3.10 (R_OH = 1.809 a.u., angle 104.52 deg), which is the geometry in
    // example/geom.xyz. Both references below round to that value; SP and DP differ
    // only in the trailing digits due to floating-point accumulation in the SCF.

    // SCF converged
    fprintf(stdout, "SCF converged! E_total = %20.10f\n", ene_total);

    const double ref_energy = use_dp ? -74.9629229672 : -74.9629154055;
    const double tol = use_dp ? 1.0e-6 : 1.0e-4;
    const double err = fabs(ene_total - ref_energy);
    if (err < tol) {
        fprintf(stdout, "PASS: E_total error %.2e within tolerance %.0e\n", err, tol);
        // print MO information
        fprintf(stdout, "%5s %10s %15s %12s\n", "MO", "State", "E(Eh)", "E(eV)");
        for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
        {
            char occ[10];
            if (ibasis < n_occ) { strcpy(occ, "occ."); }
            else { strcpy(occ, "virt."); }

            double ener = gsl_vector_get(emo, ibasis);
            fprintf(stdout, "%5d %10s %15.5f %12.2f\n",
                    ibasis + 1, occ, ener, ener * HARTREE2EV);
        }
    } else {
        fprintf(stderr, "FAIL: E_total = %.10f, expected %.10f (error %.2e > tol %.0e)\n",
                ene_total, ref_energy, err, tol);
        status = 1;
    }

    start = std::chrono::steady_clock::now();

  } // omp target

    //====== free host memories ========

    free(h_mat_D);
    free(h_mat_Q);
    free(h_mat_J);
    free(h_mat_K);

    // free DIIS error and Fock matrices
    for (idiis = 0; idiis < MAX_DIIS_DIM; ++ idiis)
    {
        for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
        {
            free(diis_err[idiis][ibasis]);
            free(diis_Fock[idiis][ibasis]);
        }
        free(diis_err[idiis]);
        free(diis_Fock[idiis]);
    }
    free(diis_err);
    free(diis_Fock);

    // free arrays for one- and two-electron integral
    gsl_matrix_free(S);
    gsl_matrix_free(T);
    gsl_matrix_free(V);

    // free matrices and vector for SCF
    gsl_matrix_free(H_core);
    gsl_matrix_free(S_invsqrt);
    gsl_matrix_free(D_prev);
    gsl_matrix_free(Fock);
    gsl_matrix_free(Coef);
    gsl_matrix_free(D);
    gsl_vector_free(emo);

    gsl_matrix_free(D_diff);
    gsl_matrix_free(Fock_prev);

    gsl_matrix_free(J);
    gsl_matrix_free(K);

    // free arrays for geometry
    for (int iatom = 0; iatom < p_atom->num; ++ iatom)
    {
        free(p_atom->pos[iatom]);
        free(p_atom->name[iatom]);
    }
    free(p_atom->pos);
    free(p_atom->name);

    free(p_atom->nuc_chg);

    free(p_atom);

    // free arrays for basis set
    for (ibasis = 0; ibasis < p_basis->num; ++ ibasis)
    {
        free(p_basis->expon[ibasis]);
        free(p_basis->coef[ibasis]);
        free(p_basis->lx[ibasis]);
        free(p_basis->ly[ibasis]);
        free(p_basis->lz[ibasis]);
        free(p_basis->norm[ibasis]);
    }
    free(p_basis->expon);
    free(p_basis->coef);
    free(p_basis->lx);
    free(p_basis->ly);
    free(p_basis->lz);
    free(p_basis->norm);

    free(p_basis->xbas);
    free(p_basis->ybas);
    free(p_basis->zbas);

    free(p_basis->nprims);

    free(p_basis);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_in_usec = time * 1e-3f;
    time_txt += "Time_Finalize = " + std::to_string(time_in_usec) + " usec\n";
    time_total += time_in_usec;

    std::cout << time_txt;
    std::cout << "Total time: " << time_total << " usec\n";

    std::cout << "Total kernel Mat_J time: " << time_mat_J << " usec\n";
    std::cout << "Total kernel Mat_K time: " << time_mat_K << " usec\n";


    //====== the end of program ========

    return status;
}
