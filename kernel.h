/*
    optimized SIMD pooling kernel
*/

#include <immintrin.h>

void print_simd(float * cptr, int size){
  printf("\t[");
  for (int i = 0; i != size; ++i){
    if (i == size - 1)
      printf("%.2f", cptr[i]);
    else
      printf("%.2f\t", cptr[i]);
  }
  printf("]\n");
}


void kernel // 
(
    float*      restrict output,    // pointer to the output matrix
    float*      restrict input,     // pointer to the input matrix
    int         k,                  // number of layers
    int         m,                  // number of rows per layer
    int         n,                  // number of columns per layer
    int         pool                // pooling size
) {

    int num_blocks = k / 8;          //  number of blocks of 8 layers
    int m_out = m / pool;            
    int n_out = n / pool;            


    for (int block = 0; block < num_blocks; block++) { // within a 8 * m * n block 

        for (int i = 0; i != m_out; ++i) {

            for (int j = 0; j != n_out; ++j) {
                
                __m256 max_vector, current_vector;

                 for (int x = 0; x < pool; x++) {

                    for (int y = 0; y < pool; y++) {

                        if (x == 0 && y == 0){
                            max_vector = _mm256_load_ps(&input[8 * (block * m * n + (i * pool + x) * n + j * pool + y)]);
                        }
                        else{
                            current_vector = _mm256_load_ps(&input[8 * (block * m * n + (i * pool + x) * n + j * pool + y)]);
                            max_vector = _mm256_max_ps(max_vector, current_vector);
                        }

                    }

                }

                _mm256_store_ps(&output[8 * (block * m_out * n_out + i * n_out + j)], max_vector);
            }
        }
    }

}