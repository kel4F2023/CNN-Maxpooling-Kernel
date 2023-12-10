#ifndef UTILS_H
#define UTILS_H

void print_col(float * cptr, int size);
void print_matrix(float * cptr, int m, int n);
void print_3d_matrix(float * cptr, int k, int m, int n);
void print_m256d(__m256 v);
static __inline__ unsigned long long rdtsc(void);
int compare_matrix (float * a, float * b, int size);

#endif
