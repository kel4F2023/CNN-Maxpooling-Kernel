#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <mpi.h>

#include "pack.h"
#include "pooling.h"
#include "utils.h"

#define RUNS 1000
#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define VERBOSE 1
#define CORRECTNESS_CHECK 1

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

int main(int argc, char** argv){

  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  float *source, *pack_source;
  float *output, *output_check, *pack_output;

  int m = 2;      // m is the number of rows per layer 
  int n = 2;      // n is the number of columns per layer 
  int k = 32;      // k is the number of layers (multiple of 32)
  int pool = 2;   // pool is the size of the pooling window

  int z = m * n;  // z is the size of a layer (m * n)
  int print_processor = 0; //who prints their data.

  if (VERBOSE && world_rank == print_processor) {
    printf("\n\n = = = = = = VERBOSE = = processor %d = = = = \n\n", world_rank);
    printf("The experiment parameters are: \n\n");
    printf("    input matrix  = %d * %d * %d = %d\n", k, m, n, k*m*n);
    printf("    output matrix = %d * %d * %d = %d\n", k, m/pool, n/pool, k*m*n/pool/pool);
    printf("\n");
  }

  /*
    Assume the following
        - All matrices are storedin row major order
  */

  posix_memalign((void**) &source, 64, z * k * sizeof(float));
  posix_memalign((void**) &pack_source, 64, z * k * sizeof(float));
  posix_memalign((void**) &output, 64, m * n / pool / pool * k * sizeof(float));
  posix_memalign((void**) &pack_output, 64, m * n / pool / pool * k * sizeof(float));
  posix_memalign((void**) &output_check, 64, m * n / pool / pool * k * sizeof(float));


  //initialize input matrix 
  for (int i = 0; i != k * z; ++i){
    source[i] = ((float) rand())/ ((float) RAND_MAX);
    pack_source[i] = 0;
  }

  if (VERBOSE && world_rank == print_processor) {
    printf("The input matrix is: \n");
    // print_3d_matrix(source, k, m, n);
    print_matrix(source, k, m*n);
    printf("\n");
  }

  pack(pack_source, source, m, n, k, pool);

  if (VERBOSE && world_rank == print_processor) {
    printf("The packed matrix is: \n");
    // print_3d_matrix(pack_source, m, n, k);
    print_matrix(pack_source, m*n, k);
    printf("\n");
  }

  kernel(pack_output, pack_source, k, m, n, pool);

  if (VERBOSE && world_rank == print_processor) {
    printf("The packed output matrix is: \n");
    // print_3d_matrix(output, k, m/pool, n/pool);
    print_matrix(pack_output, k*m*n/pool/pool/32, 32);
    printf("\n");
  }

  unpack(output, pack_output, m*n/pool/pool, k);

  if (VERBOSE && world_rank == print_processor) {
    printf("The output matrix is: \n");
    // print_3d_matrix(output, k, m/pool, n/pool);
    print_matrix(output, k*m*n/pool/pool/32, 32);
    printf("\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (CORRECTNESS_CHECK) {

    naive(output_check, source, k, m, n, pool);

    if (VERBOSE && world_rank == print_processor) {
      printf("The correct output matrix is: \n");
      // print_3d_matrix(output_check, k, m/pool, n/pool);
      print_matrix(output_check, k*m*n/pool/pool/32, 32);
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (compare_matrix(output, output_check, k*m*n/pool/pool)) {
      printf("processor %d is correct!\n", world_rank);
    } else {
      printf("processor %d is incorrect!\n", world_rank);
    }
  }

  free(source);
  free(pack_source);
  free(output);
  free(pack_output);
  free(output_check);

  MPI_Finalize();
  
  return 0;
}