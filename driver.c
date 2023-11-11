#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#include "kernel.h"
#include "pack.h"
#include "naive.h"
#include "utils.h"

#define RUNS 1000
#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define VERBOSE 0
#define CORRECTNESS_CHECK 1

int main(){

  float *source, *pack_source;
  float *output, *output_check, *pack_output;

  unsigned long long t0, t1, sum;

  int m = 8;      // m is the number of rows per layer 
  int n = 8;      // n is the number of columns per layer 
  int k = 64;      // k is the number of layers
  int pool = 4;   // pool is the size of the pooling window

  int z = m * n;  // z is the size of a layer (m * n)

  if (VERBOSE) {
    printf("\n\n = = = = = = VERBOSE = = = = = = = \n\n");
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

  if (VERBOSE) {
    printf("The input matrix is: \n");
    // print_3d_matrix(source, k, m, n);
    print_matrix(source, k, m*n);
    printf("\n");
  }

  pack(pack_source, source, m, n, k, pool);

  if (VERBOSE) {
    printf("The packed matrix is: \n");
    // print_3d_matrix(pack_source, m, n, k);
    print_matrix(pack_source, m*n, k);
    printf("\n");
  }

  kernel(pack_output, pack_source, k, m, n, pool);

  if (VERBOSE) {
    printf("The packed output matrix is: \n");
    // print_3d_matrix(output, k, m/pool, n/pool);
    print_matrix(pack_output, k*m*n/pool/pool/32, 32);
    printf("\n");
  }

  unpack(output, pack_output, m*n/pool/pool, k);

  if (VERBOSE) {
    printf("The output matrix is: \n");
    // print_3d_matrix(output, k, m/pool, n/pool);
    print_matrix(output, k*m*n/pool/pool/32, 32);
    printf("\n");
  }

  if (CORRECTNESS_CHECK) {

    naive(output_check, source, k, m, n, pool);

    if (VERBOSE) {
      printf("The correct output matrix is: \n");
      // print_3d_matrix(output_check, k, m/pool, n/pool);
      print_matrix(output_check, k*m*n/pool/pool/32, 32);
      printf("\n");
    }

    if (compare_matrix(output, output_check, k*m*n/pool/pool)) {
      printf("The output matrix is correct!\n");
    } else {
      printf("The output matrix is incorrect!\n");
    }
  }

  free(source);
  free(pack_source);
  free(output);
  free(pack_output);
  free(output_check);
  
  return 0;
}