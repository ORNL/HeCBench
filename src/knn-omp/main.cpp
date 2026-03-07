/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 *
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

// Includes
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

// Constants used by the program
#define BLOCK_DIM 16

//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS //
//-----------------------------------------------------------------------------------------------//

float compute_distance(const float *ref, int ref_nb, const float *query,
                       int query_nb, int dim, int ref_index, int query_index) {
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float diff =
        ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

void modified_insertion_sort(float *dist, int *index, int length, int k) {

  // Initialise the first index
  index[0] = 0;

  // Go through all points
  for (int i = 1; i < length; ++i) {

    // Store current distance and associated index
    float curr_dist = dist[i];
    int curr_index = i;

    // Skip the current value if its index is >= k and if it's higher the k-th
    // slready sorted mallest value
    if (i >= k && curr_dist >= dist[k - 1]) {
      continue;
    }

    // Shift values (and indexes) higher that the current distance to the right
    int j = i < k - 1 ? i : k-1;
    while (j > 0 && dist[j - 1] > curr_dist) {
      dist[j] = dist[j - 1];
      index[j] = index[j - 1];
      --j;
    }

    // Write the current distance and index at their position
    dist[j] = curr_dist;
    index[j] = curr_index;
  }
}

bool knn_serial(const float *ref, int ref_nb, const float *query, int query_nb,
           int dim, int k, float *knn_dist, int *knn_index) {
  // Allocate local array to store all the distances / indexes for a given query
  // point
  float *dist = (float *)malloc(ref_nb * sizeof(float));
  int *index = (int *)malloc(ref_nb * sizeof(int));

  // Allocation checks
  if (!dist || !index) {
    printf("Memory allocation error\n");
    free(dist);
    free(index);
    return false;
  }

  // Process one query point at the time
  for (int i = 0; i < query_nb; ++i) {

    // Compute all distances / indexes
    for (int j = 0; j < ref_nb; ++j) {
      dist[j] = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
      index[j] = j;
    }

    // Sort distances / indexes
    modified_insertion_sort(dist, index, ref_nb, k);

    // Copy k smallest distances and their associated index
    for (int j = 0; j < k; ++j) {
      knn_dist[j * query_nb + i] = dist[j];
      knn_index[j * query_nb + i] = index[j];
    }
  }

  // Memory clean-up
  free(dist);
  free(index);
  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int iterations = atoi(argv[1]);

  float *ref;          // Pointer to reference point array
  float *query;        // Pointer to query point array
  float *dist;         // Pointer to distance array
  int *ind;            // Pointer to index array
  int ref_nb = 4096;   // Reference point number, max=65535
  int query_nb = 4096; // Query point number,     max=65535
  int dim = 68;        // Dimension of points
  int k = 20;          // Nearest neighbors to consider
  int c_iterations = 1;
  int i;
  const float precision = 0.001f; // distance error max
  int nb_correct_precisions = 0;
  int nb_correct_indexes = 0;
  // Memory allocation
  ref = (float *)malloc(ref_nb * dim * sizeof(float));
  query = (float *)malloc(query_nb * dim * sizeof(float));
  //dist = (float *)malloc(query_nb * k * sizeof(float));
  dist = (float *)malloc(query_nb * ref_nb * sizeof(float));
  ind = (int *)malloc(query_nb * k * sizeof(float));

  // Init
  srand(2);
  for (i = 0; i < ref_nb * dim; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dim; i++)
    query[i] = (float)rand() / (float)RAND_MAX;

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float *knn_dist = (float *)malloc(query_nb * k * sizeof(float));
  int *knn_index = (int *)malloc(query_nb * k * sizeof(int));
  printf("Ground truth computation in progress...\n\n");
  if (!knn_serial(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
    return EXIT_FAILURE;
  }

  struct timeval tic;
  struct timeval toc;
  float elapsed_time;

  printf("On CPU: \n");
  gettimeofday(&tic, NULL);
  for (i = 0; i < c_iterations; i++) {
    knn_serial(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         c_iterations, elapsed_time / (c_iterations));

  printf("on GPU: \n");
  gettimeofday(&tic, NULL);

  for (i = 0; i < iterations; i++) {
    #pragma omp target data map(to: ref[0:ref_nb * dim], query[0:query_nb * dim]) \
                          map(alloc: dist[0:query_nb * ref_nb], ind[0:query_nb * k])
    {
      // Kernel 1: Compute all the distances
      #pragma omp target teams num_teams((ref_nb + 15)*(query_nb + 15)/256) thread_limit(256) 
      {
        float shared_A[BLOCK_DIM*BLOCK_DIM];
        float shared_B[BLOCK_DIM*BLOCK_DIM];
        int begin_A;
        int begin_B;
        int step_A;
        int step_B;
        int end_A;
        
        #pragma omp parallel 
        {
          // Thread index
          int tx = omp_get_thread_num() % 16;
          int ty = omp_get_thread_num() / 16;
      
          // Other variables
          float tmp;
          float ssd = 0;
      
          // Loop parameters
          begin_A = BLOCK_DIM * (omp_get_team_num() / ((query_nb+15)/16));
          begin_B = BLOCK_DIM * (omp_get_team_num() % ((query_nb+15)/16));
          step_A  = BLOCK_DIM * ref_nb;
          step_B  = BLOCK_DIM * query_nb;
          end_A   = begin_A + (dim - 1) * ref_nb;
      
          // Conditions
          int cond0 = (begin_A + tx < ref_nb); // used to write in shared memory
          int cond1 = (begin_B + tx < query_nb); // used to write in shared memory & to
                                           // computations and to write in output matrix
          int cond2 =
              (begin_A + ty < ref_nb); // used to computations and to write in output matrix
      
          // Loop over all the sub-matrices of A and B required to compute the block
          // sub-matrix
          for (int a = begin_A, b = begin_B; 
                   a <= end_A; a += step_A, b += step_B) {
            // Load the matrices from device memory to shared memory; each thread loads
            // one element of each matrix
            if (a / ref_nb + ty < dim) {
              shared_A[ty*BLOCK_DIM+tx] = (cond0) ? ref[a + ref_nb * ty + tx] : 0;
              shared_B[ty*BLOCK_DIM+tx] = (cond1) ? query[b + query_nb * ty + tx] : 0;
            } else {
              shared_A[ty*BLOCK_DIM+tx] = 0;
              shared_B[ty*BLOCK_DIM+tx] = 0;
            }
      
            // Synchronize to make sure the matrices are loaded
            #pragma omp barrier
      
            // Compute the difference between the two matrixes; each thread computes one
            // element of the block sub-matrix
            if (cond2 && cond1) {
              for (int k = 0; k < BLOCK_DIM; ++k) {
                tmp = shared_A[k*BLOCK_DIM+ty] - shared_B[k*BLOCK_DIM+tx];
                ssd += tmp * tmp;
              }
            }
      
            // Synchronize to make sure that the preceding computation is done before
            // loading two new sub-matrices of A and B in the next iteration
            #pragma omp barrier
          }
      
          // Write the block sub-matrix to device memory; each thread writes one element
          if (cond2 && cond1) dist[(begin_A + ty) * query_nb + begin_B + tx] = ssd;
        }
      }
      
      // Kernel 2: Sort each column
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (unsigned int xIndex = 0; xIndex < query_nb; xIndex++) {
        // Pointer shift, initialization, and max value
        float* p_dist = &dist[xIndex];
        int* p_ind = &ind[xIndex];
        float max_dist = p_dist[0];
        p_ind[0] = 0;
      
        // Part 1 : sort kth firt elementZ
        for (int l = 1; l < k; l++) {
          int curr_row = l * query_nb;
          float curr_dist = p_dist[curr_row];
          if (curr_dist < max_dist) {
            int i = l - 1;
            for (int a = 0; a < l - 1; a++) {
              if (p_dist[a * query_nb] > curr_dist) {
                i = a;
                break;
              }
            }
            for (int j = l; j > i; j--) {
              p_dist[j * query_nb] = p_dist[(j - 1) * query_nb];
              p_ind[j * query_nb] = p_ind[(j - 1) * query_nb];
            }
            p_dist[i * query_nb] = curr_dist;
            p_ind[i * query_nb] = l;
          } else {
            p_ind[l * query_nb] = l;
          }
          max_dist = p_dist[curr_row];
        }
      
        // Part 2 : insert element in the k-th first lines
        int max_row = (k - 1) * query_nb;
        for (int l = k; l < ref_nb; l++) {
          float curr_dist = p_dist[l * query_nb];
          if (curr_dist < max_dist) {
            int i = k - 1;
            for (int a = 0; a < k - 1; a++) {
              if (p_dist[a * query_nb] > curr_dist) {
                i = a;
                break;
              }
            }
            for (int j = k - 1; j > i; j--) {
              p_dist[j * query_nb] = p_dist[(j - 1) * query_nb];
              p_ind[j * query_nb] = p_ind[(j - 1) * query_nb];
            }
            p_dist[i * query_nb] = curr_dist;
            p_ind[i * query_nb] = l;
            max_dist = p_dist[max_row];
          }
        }
      }
      
      // Kernel 3: Compute square root of k first elements
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (unsigned int i = 0; i < query_nb * k; i++)
        dist[i] = sqrtf(dist[i]);

      #pragma omp target update from (dist[0:query_nb * k]) 
      #pragma omp target update from (ind[0:query_nb * k])
    }
  }

  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         iterations, elapsed_time / (iterations));

  for (int i = 0; i < query_nb * k; ++i) {
    if (fabs(dist[i] - knn_dist[i]) <= precision) {
      nb_correct_precisions++;
    }
    if (ind[i] == knn_index[i]) {
      nb_correct_indexes++;
    }
  }

  float precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  float index_accuracy = nb_correct_indexes / ((float)query_nb * k);
  printf("Precision accuracy %f\nIndex accuracy %f\n", precision_accuracy, index_accuracy);
  printf("%s\n", (precision_accuracy == 1.f) ? "PASS" : "FAIL");
  if (!(precision_accuracy == 1.f)) exit(1);

  free(ind);
  free(dist);
  free(query);
  free(ref);
}
