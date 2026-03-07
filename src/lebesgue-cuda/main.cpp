#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include "lebesgue.h"
#include "reference.h"

//
// The expected test results are shown in 
// https://people.math.sc.edu/Burkardt/c_src/lebesgue/lebesgue_test.txt
// 

void test01 ( int nfun  )
{
  //char label[] = "Chebyshev1 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST01:\n" );
  printf ( "  Analyze Chebyshev1 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = chebyshev1 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l,
    "  Chebyshev1 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = chebyshev1 ( n );
  r8vec_print ( n, x, "  Chebyshev1 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test02 ( int nfun  )
{
  //char label[] = "Chebyshev2 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST02:\n" );
  printf ( "  Analyze Chebyshev2 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = chebyshev2 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l,
    "  Chebyshev2 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = chebyshev2 ( n );
  r8vec_print ( n, x, "  Chebyshev2 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test03 ( int nfun  )
{
  // char label[] = "Chebyshev3 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST03:\n" );
  printf ( "  Analyze Chebyshev3 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = chebyshev3 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l, 
    "  Chebyshev3 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = chebyshev3 ( n );
  r8vec_print ( n, x, "  Chebyshev3 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test04 ( int nfun  )
{
  //char label[] = "Chebyshev4 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST04:\n" );
  printf ( "  Analyze Chebyshev4 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = chebyshev4 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l, 
    "  Chebyshev4 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = chebyshev4 ( n );
  r8vec_print ( n, x, "  Chebyshev4 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test05 ( int nfun  )
{
  //char label[] = "Equidistant1 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST05:\n" );
  printf ( "  Analyze Equidistant1 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = equidistant1 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l, 
    "  Equidistant1 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = equidistant1 ( n );
  r8vec_print ( n, x, "  Equidistant1 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test06 ( int nfun  )
{
  //char label[] = "Equidistant2 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST06:\n" );
  printf ( "  Analyze Equidistant2 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = equidistant2 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l, 
    "  Equidistant2 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = equidistant2 ( n );
  r8vec_print ( n, x, "  Equidistant2 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test07 ( int nfun  )
{
  //char label[] = "Equidistant3 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST07:\n" );
  printf ( "  Analyze Equidistant3 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = equidistant3 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l,
    "  Equidistant3 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = equidistant3 ( n );
  r8vec_print ( n, x, "  Equidistant3 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test08 ( int nfun  )
{
  //char label[] = "Fejer1 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST08:\n" );
  printf ( "  Analyze Fejer1 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = fejer1 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l,
    "  Fejer1 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = fejer1 ( n );
  r8vec_print ( n, x, "  Fejer1 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

void test09 ( int nfun  )
{
  //char label[] = "Fejer2 points for N = 11";
  double *l;
  int n;
  int n_max = 11;
  double *x;
  double *xfun;

  printf ( "\n" );
  printf ( "LEBESGUE_TEST09:\n" );
  printf ( "  Analyze Fejer2 points.\n" );

  xfun = r8vec_linspace_new ( nfun, -1.0, +1.0 );

  l = ( double * ) malloc ( n_max * sizeof ( double ) );

  float total_time = 0.f;
  bool ok = true;

  for ( n = 1; n <= n_max; n++ )
  {
    x = fejer2 ( n );

    auto start = std::chrono::steady_clock::now();

    l[n-1] = lebesgue_constant ( n, x, nfun, xfun );

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    ok &= verify(l[n-1], n, x, nfun, xfun);
    free ( x );
  }
  printf ( "  Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf ( "  %s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifdef DEBUG
  r8vec_print ( n_max, l,
    "  Fejer2 Lebesgue constants for N = 1 to 11:" );
/*
  Examine one case more closely.
*/
  n = 11;
  x = fejer2 ( n );
  r8vec_print ( n, x, "  Fejer2 points for N = 11" );
  free ( x );
#endif

  free ( l );
  free ( xfun );
}

int main (int argc, char* argv[] )
{
  if (argc != 3) {
    printf("Usage: %s <number of points in an interval> <repeat>\n", argv[0]);
    return 1;
  }
  int nfun = atoi(argv[1]);
  int repeat = atoi(argv[2]);

  printf ( "\n" );
  printf ( "LEBESGUE_TEST\n" );

  for (int i = 0; i < repeat; i++) {
    timestamp ( );

    test01 ( nfun );
    test02 ( nfun );
    test03 ( nfun );
    test04 ( nfun );
    test05 ( nfun );
    test06 ( nfun );
    test07 ( nfun );
    test08 ( nfun );
    test09 ( nfun );

    timestamp ( );
  }

  return 0;
}
