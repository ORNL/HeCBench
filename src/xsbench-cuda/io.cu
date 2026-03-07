#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

// Prints Section titles in center of 80 char terminal
void center_print(const char *s, int width)
{
  int length = strlen(s);
  int i;
  for (i=0; i<=(width-length)/2; i++) {
    fputs(" ", stdout);
  }
  fputs(s, stdout);
  fputs("\n", stdout);
}

int print_results( Inputs in, int mype, double runtime, int nprocs,
                   unsigned long long *vhash,
                   double kernel_time )
{
  // Calculate Lookups per sec
  int lookups = 0;
  if( in.simulation_method == HISTORY_BASED )
    lookups = in.lookups * in.particles;
  else if( in.simulation_method == EVENT_BASED )
    lookups = in.lookups;
  int lookups_per_sec = (int) ((double) lookups / runtime);
  int sim_only_lookups_per_sec = (int) ((double) lookups/ (kernel_time));

  // If running in MPI, reduce timing statistics and calculate average
#ifdef MPI
  int total_lookups = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&lookups_per_sec, &total_lookups, 1, MPI_INT,
      MPI_SUM, 0, MPI_COMM_WORLD);
#endif

  // Print output
  if( mype == 0 )
  {
    border_print();
    center_print("RESULTS", 79);
    border_print();

    // Print the results
#ifdef MPI
    printf("MPI ranks:   %d\n", nprocs);
#endif
#ifdef MPI
    printf("Total Lookups/s:            ");
    fancy_int(total_lookups);
    printf("Avg Lookups/s per MPI rank: ");
    fancy_int(total_lookups / nprocs);
#else
    printf("Total Time Statistics (Device Init / JIT Compilation + Simulation Kernel)\n");
    printf("Runtime:               %.3lf seconds\n", runtime);
    printf("Lookups:               ");fancy_int(lookups);
    printf("Lookups/s:             ");
    fancy_int(lookups_per_sec);
    printf("Simulation Kernel Only Statistics\n");
    printf("Average kernel execution time: %.3lf seconds\n", kernel_time);
    printf("Lookups/s:             ");
    fancy_int(sim_only_lookups_per_sec);
#endif
  }
  int is_invalid_result = vhash[0] != vhash[1];

  if(mype == 0 )
  {
    if( is_invalid_result )
      printf("Verification checksum: %llu != %llu (WARNING - INVALID CHECKSUM!)\n", vhash[0], vhash[1]);
    else
      printf("Verification checksum: %llu (Valid)\n", vhash[0]);
    border_print();
  }

  return is_invalid_result;
}

void print_inputs(Inputs in, int nprocs, int version )
{
  // Calculate Estimate of Memory Usage
  int mem_tot = estimate_mem_usage( in );
  center_print("INPUT SUMMARY", 79);
  border_print();
  if( in.simulation_method == EVENT_BASED )
    printf("Simulation Method:            Event Based\n");
  else
    printf("Simulation Method:            History Based\n");
  if( in.grid_type == NUCLIDE )
    printf("Grid Type:                    Nuclide Grid\n");
  else if( in.grid_type == UNIONIZED )
    printf("Grid Type:                    Unionized Grid\n");
  else
    printf("Grid Type:                    Hash\n");

  printf("Materials:                    %d\n", 12);
  printf("H-M Benchmark Size:           %s\n", in.HM);
  printf("Total Nuclides:               %ld\n", in.n_isotopes);
  printf("Gridpoints (per Nuclide):     ");
  fancy_int(in.n_gridpoints);
  if( in.grid_type == HASH )
  {
    printf("Hash Bins:                    ");
    fancy_int(in.hash_bins);
  }
  if( in.grid_type == UNIONIZED )
  {
    printf("Unionized Energy Gridpoints:  ");
    fancy_int(in.n_isotopes*in.n_gridpoints);
  }
  if( in.simulation_method == HISTORY_BASED )
  {
    printf("Particle Histories:           "); fancy_int(in.particles);
    printf("XS Lookups per Particle:      "); fancy_int(in.lookups);
  }
  printf("Total XS Lookups:             "); fancy_int(in.lookups);
#ifdef MPI
  printf("MPI Ranks:                    %d\n", nprocs);
  printf("Mem Usage per MPI Rank (MB):  "); fancy_int(mem_tot);
#else
  printf("Est. Memory Usage (MB):       "); fancy_int(mem_tot);
#endif
  printf("Binary File Mode:             ");
  if( in.binary_mode == NONE )
    printf("Off\n");
  else if( in.binary_mode == READ)
    printf("Read\n");
  else
    printf("Write\n");
  border_print();
  center_print("INITIALIZATION - DO NOT PROFILE", 79);
  border_print();
}

void border_print(void)
{
  printf(
      "==================================================================="
      "=============\n");
}

// Prints comma separated integers - for ease of reading
void fancy_int( long a )
{
  if( a < 1000 )
    printf("%ld\n",a);

  else if( a >= 1000 && a < 1000000 )
    printf("%ld,%03ld\n", a / 1000, a % 1000);

  else if( a >= 1000000 && a < 1000000000 )
    printf("%ld,%03ld,%03ld\n",a / 1000000,(a % 1000000) / 1000,a % 1000 );

  else if( a >= 1000000000 )
    printf("%ld,%03ld,%03ld,%03ld\n",
        a / 1000000000,
        (a % 1000000000) / 1000000,
        (a % 1000000) / 1000,
        a % 1000 );
  else
    printf("%ld\n",a);
}

void print_CLI_error(void)
{
  printf("Usage: ./main <options>\n");
  printf("Options include:\n");
  printf("  -m <simulation method>   Simulation method (history, event)\n");
  printf("  -s <size>                Size of H-M Benchmark to run (small, large, XL, XXL)\n");
  printf("  -g <gridpoints>          Number of gridpoints per nuclide (overrides -s defaults)\n");
  printf("  -G <grid type>           Grid search type (unionized, nuclide, hash). Defaults to unionized.\n");
  printf("  -p <particles>           Number of particle histories\n");
  printf("  -l <lookups>             History Based: Number of Cross-section (XS) lookups per particle. Event Based: Total number of XS lookups.\n");
  printf("  -h <hash bins>           Number of hash bins (only relevant when used with \"-G hash\")\n");
  printf("  -b <binary mode>         Read or write all data structures to file. If reading, this will skip initialization phase. (read, write)\n");
  printf("  -k <kernel ID>           Specifies which kernel to run. 0 is baseline, 1, 2, etc are optimized variants. (0 is default.)\n");
  printf("  -r <kernel count>        Specifies the kernel execution count. (1 is default.)\n");
  printf("Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized\n");
  printf("See readme for full description of default run values\n");
  exit(4);
}

Inputs read_CLI( int argc, char * argv[] )
{
  Inputs input;

  // defaults to the history based simulation method
  input.simulation_method = HISTORY_BASED;

  // defaults to max threads on the system  
  input.nthreads = 1;

  // defaults to 355 (corresponding to H-M Large benchmark)
  input.n_isotopes = 355;

  // defaults to 11303 (corresponding to H-M Large benchmark)
  input.n_gridpoints = 11303;

  // defaults to 500,000
  input.particles = 500000;

  // defaults to 34
  input.lookups = 34;

  // default to unionized grid
  input.grid_type = UNIONIZED;

  // default to unionized grid
  input.hash_bins = 10000;

  // default to no binary read/write
  input.binary_mode = NONE;

  // defaults to baseline kernel
  input.kernel_id = 0;

  // defaults to kernel execution once 
  input.kernel_repeat = 1;

  // defaults to H-M Large benchmark
  input.HM = (char *) malloc( 6 * sizeof(char) );
  input.HM[0] = 'l' ; 
  input.HM[1] = 'a' ; 
  input.HM[2] = 'r' ; 
  input.HM[3] = 'g' ; 
  input.HM[4] = 'e' ; 
  input.HM[5] = '\0';

  // Check if user sets these
  int user_g = 0;

  int default_lookups = 1;
  int default_particles = 1;

  // Collect Raw Input
  for( int i = 1; i < argc; i++ )
  {
    char * arg = argv[i];

    // n_gridpoints (-g)
    if( strcmp(arg, "-g") == 0 )
    {  
      if( ++i < argc )
      {
        user_g = 1;
        input.n_gridpoints = atol(argv[i]);
      }
      else
        print_CLI_error();
    }
    // Simulation Method (-m)
    else if( strcmp(arg, "-m") == 0 )
    {
      char * sim_type;
      if( ++i < argc )
        sim_type = argv[i];
      else
        print_CLI_error();

      if( strcmp(sim_type, "history") == 0 )
        input.simulation_method = HISTORY_BASED;
      else if( strcmp(sim_type, "event") == 0 )
      {
        input.simulation_method = EVENT_BASED;
        // Also resets default # of lookups
        if( default_lookups && default_particles )
        {
          input.lookups =  input.lookups * input.particles;
          input.particles = 0;
        }
      }
      else
        print_CLI_error();
    }
    // lookups (-l)
    else if( strcmp(arg, "-l") == 0 )
    {
      if( ++i < argc )
      {
        input.lookups = atoi(argv[i]);
        default_lookups = 0;
      }
      else
        print_CLI_error();
    }
    // hash bins (-h)
    else if( strcmp(arg, "-h") == 0 )
    {
      if( ++i < argc )
        input.hash_bins = atoi(argv[i]);
      else
        print_CLI_error();
    }
    // particles (-p)
    else if( strcmp(arg, "-p") == 0 )
    {
      if( ++i < argc )
      {
        input.particles = atoi(argv[i]);
        default_particles = 0;
      }
      else
        print_CLI_error();
    }
    // HM (-s)
    else if( strcmp(arg, "-s") == 0 )
    {  
      if( ++i < argc )
        input.HM = argv[i];
      else
        print_CLI_error();
    }
    // grid type (-G)
    else if( strcmp(arg, "-G") == 0 )
    {
      char * grid_type;
      if( ++i < argc )
        grid_type = argv[i];
      else
        print_CLI_error();

      if( strcmp(grid_type, "unionized") == 0 )
        input.grid_type = UNIONIZED;
      else if( strcmp(grid_type, "nuclide") == 0 )
        input.grid_type = NUCLIDE;
      else if( strcmp(grid_type, "hash") == 0 )
        input.grid_type = HASH;
      else
        print_CLI_error();
    }
    // binary mode (-b)
    else if( strcmp(arg, "-b") == 0 )
    {
      char * binary_mode;
      if( ++i < argc )
        binary_mode = argv[i];
      else
        print_CLI_error();

      if( strcmp(binary_mode, "read") == 0 )
        input.binary_mode = READ;
      else if( strcmp(binary_mode, "write") == 0 )
        input.binary_mode = WRITE;
      else
        print_CLI_error();
    }
    // kernel optimization selection (-k)
    else if( strcmp(arg, "-k") == 0 )
    {
      if( ++i < argc )
      {
        input.kernel_id = atoi(argv[i]);
      }
      else
        print_CLI_error();
    }
    // kernel count (-r)
    else if( strcmp(arg, "-r") == 0 )
    {
      if( ++i < argc )
      {
        input.kernel_repeat = atoi(argv[i]);
      }
      else
        print_CLI_error();
    }
    else
      print_CLI_error();
  }

  // Validate Input

  // Validate nthreads
  if( input.nthreads < 1 )
    print_CLI_error();

  // Validate n_isotopes
  if( input.n_isotopes < 1 )
    print_CLI_error();

  // Validate n_gridpoints
  if( input.n_gridpoints < 1 )
    print_CLI_error();

  // Validate lookups
  if( input.lookups < 1 )
    print_CLI_error();

  // Validate Hash Bins 
  if( input.hash_bins < 1 )
    print_CLI_error();

  // Validate HM size
  if( strcasecmp(input.HM, "small") != 0 &&
      strcasecmp(input.HM, "large") != 0 &&
      strcasecmp(input.HM, "XL") != 0 &&
      strcasecmp(input.HM, "XXL") != 0 )
    print_CLI_error();

  // Set HM size specific parameters
  // (defaults to large)
  if( strcasecmp(input.HM, "small") == 0 )
    input.n_isotopes = 68;
  else if( strcasecmp(input.HM, "XL") == 0 && user_g == 0 )
    input.n_gridpoints = 238847; // sized to make 120 GB XS data
  else if( strcasecmp(input.HM, "XXL") == 0 && user_g == 0 )
    input.n_gridpoints = 238847 * 2.1; // 252 GB XS data

  // Return input struct
  return input;
}

void binary_write( Inputs in, SimulationData SD )
{
  const char * fname = "XS_data.dat";
  printf("Writing all data structures to binary file %s...\n", fname);
  FILE * fp = fopen(fname, "w");

  // Write SimulationData Object. Include pointers, even though we won't be using them.
  fwrite(&SD, sizeof(SimulationData), 1, fp);

  // Write heap arrays in SimulationData Object
  fwrite(SD.num_nucs,       sizeof(int), SD.length_num_nucs, fp);
  fwrite(SD.concs,          sizeof(double), SD.length_concs, fp);
  fwrite(SD.mats,           sizeof(int), SD.length_mats, fp);
  fwrite(SD.nuclide_grid,   sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp); 
  fwrite(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
  fwrite(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);

  fclose(fp);
}

SimulationData binary_read( Inputs in )
{
  SimulationData SD;

  const char * fname = "XS_data.dat";
  printf("Reading all data structures from binary file %s...\n", fname);

  FILE * fp = fopen(fname, "r");
  assert(fp != NULL);

  // Read SimulationData Object. Include pointers, even though we won't be using them.
  fread(&SD, sizeof(SimulationData), 1, fp);

  // Allocate space for arrays on heap
  SD.num_nucs = (int *) malloc(SD.length_num_nucs * sizeof(int));
  SD.concs = (double *) malloc(SD.length_concs * sizeof(double));
  SD.mats = (int *) malloc(SD.length_mats * sizeof(int));
  SD.nuclide_grid = (NuclideGridPoint *) malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
  SD.index_grid = (int *) malloc( SD.length_index_grid * sizeof(int));
  SD.unionized_energy_array = (double *) malloc( SD.length_unionized_energy_array * sizeof(double));

  // Read heap arrays into SimulationData Object
  fread(SD.num_nucs,       sizeof(int), SD.length_num_nucs, fp);
  fread(SD.concs,          sizeof(double), SD.length_concs, fp);
  fread(SD.mats,           sizeof(int), SD.length_mats, fp);
  fread(SD.nuclide_grid,   sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp); 
  fread(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
  fread(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);

  fclose(fp);

  return SD;
}
