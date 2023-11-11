/*
    optimized SIMD pooling kernel
*/

#include <immintrin.h>


void kernel // 
(
    float*      restrict output,    // pointer to the output matrix
    float*      restrict input,     // pointer to the input matrix
    int         k,                  // number of layers
    int         m,                  // number of rows per layer
    int         n,                  // number of columns per layer
    int         pool                // pooling size
) {

    int num_blocks = k / 32;          //  number of blocks of 32 layers
    int m_out = m / pool;            
    int n_out = n / pool;
    
    float result[8];        

    __m256 vi0, vi1, vi2, vi3;
    __m256 vo0, vo1, vo2, vo3;

    for (int block = 0; block < num_blocks; block++) { // within a 8 * m * n block

        for (int num_pools = 0; num_pools < m_out * n_out; num_pools++) {

            for (int pool_idx = 0; pool_idx < pool * pool; ++pool_idx) {

                // actucal kernel: 4 simd vectors with 8 float values
                if (pool_idx == 0){

                    vo0 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 0]);
                    vo1 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 8]);
                    vo2 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 16]);
                    vo3 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 24]);

                }
                else{
                    vi0 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 0]);
                    vi1 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 8]);
                    vi2 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 16]);
                    vi3 = _mm256_load_ps(&input[32 * (block * m * n + num_pools * pool * pool + pool_idx) + 24]);
                    
                    vo0 = _mm256_max_ps(vo0, vi0);
                    vo1 = _mm256_max_ps(vo1, vi1);
                    vo2 = _mm256_max_ps(vo2, vi2);
                    vo3 = _mm256_max_ps(vo3, vi3);

                }
            }     

            _mm256_store_ps(&output[32 * (block * m_out * n_out + num_pools) + 0], vo0);
            _mm256_store_ps(&output[32 * (block * m_out * n_out + num_pools) + 8], vo1);
            _mm256_store_ps(&output[32 * (block * m_out * n_out + num_pools) + 16], vo2);
            _mm256_store_ps(&output[32 * (block * m_out * n_out + num_pools) + 24], vo3);
        }
    }
}