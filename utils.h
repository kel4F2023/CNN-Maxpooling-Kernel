#include <immintrin.h>
#include <stdio.h>

void print_col(float * cptr, int size){
  printf("\t[");
  for (int i = 0; i != size; ++i){
    if (i == size - 1)
      printf("%.2f", cptr[i]);
    else
      printf("%.2f\t", cptr[i]);
  }
  printf("]\n");
}

void print_matrix(float * cptr, int m, int n){
  for (int i = 0; i != m; ++i){
    print_col(cptr + i*n, n);
  }
}

void print_3d_matrix(float * cptr, int k, int m, int n){
  printf("{");
  for (int i = 0; i != k; ++i){
    printf("\n");
    print_matrix(cptr + i*m*n, m, n);
  }
  printf("}\n");
}


void print_m256d(__m256 v){
  float result[8];
  _mm256_store_ps(result, v);
  print_col(result, 8);
}

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

int compare_matrix (float * a, float * b, int size) {

  int correct = 1;

  for (int i = 0; i != size; ++i) {
    correct &= (fabs(a[i] - b[i]) < 1e-13);
  }

  return correct;
}