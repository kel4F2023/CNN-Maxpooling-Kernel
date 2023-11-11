/*
    packing routines to reordering the data for SIMD usage
*/

#include <immintrin.h>

void pack // naive packing routine
(
  float*     restrict pack_source, // result data
  float*     restrict source,      // source data
  int               z,             // size of a layer (m * n)
  int               k              // number of layers
){

    int num_blocks = k / 8;

    for (int block = 0; block < num_blocks; block++) { // within a 8 * Z block 

        for (int i = 0; i != z; ++i) {

            for (int j = 0; j != 8; ++j) {

                pack_source[block*z*8 + i*8 + j] = source[block*z*8 + j*z + i];

            }
            
        }

    }
}

void unpack ( // naive packing routine
    float*     restrict res,              // result data
    float*     restrict pack_res,         // source data in packed format
    int               z,                  // size of a layer (m * n)
    int               k                   // number of layers
){
    int num_blocks = k / 8;

    for (int block = 0; block < num_blocks; block++) { // within a 8 * Z block 

        for (int i = 0; i != z; ++i) {

            for (int j = 0; j != 8; ++j) {

                res[block*z*8 + j*z + i] = pack_res[block*z*8 + i*8 + j];

            }
            
        }

    }
}

// void simd_pack // optimized packing routine with SIMD instructions
// (
//   float*     restrict pack_source, // result data
//   float*     restrict source,      // source data
//   int               z,         // size of a layer (m * n)
//   int               k          // number of layers
// ){
//     int num_blocks = k / 8;

//     for (int block = 0; block < num_blocks; block++) {
//         // within a 8 * Z block, we have 8 * 8 elements

//         for (int idx = 0; idx < k; idx+=8) {
            
//             __m256d row0 = _mm256_loadu_ps(&a[block * 8 * z + 0 * k + idx]);
//             __m256d row1 = _mm256_loadu_ps(&a[block * 8 * z + 1 * k + idx]);
//             __m256d row2 = _mm256_loadu_ps(&a[block * 8 * z + 2 * k + idx]);
//             __m256d row3 = _mm256_loadu_ps(&a[block * 8 * z + 3 * k + idx]);
//             __m256d row4 = _mm256_loadu_ps(&a[block * 8 * z + 4 * k + idx]);
//             __m256d row5 = _mm256_loadu_ps(&a[block * 8 * z + 5 * k + idx]);
//             __m256d row6 = _mm256_loadu_ps(&a[block * 8 * z + 6 * k + idx]);  
//             __m256d row7 = _mm256_loadu_ps(&a[block * 8 * z + 7 * k + idx]);


//             __m256d row01lo = _mm256_unpacklo_pd(row0, row1);
//             __m256d row01hi = _mm256_unpackhi_pd(row0, row1);

//             __m256d comb2row01lo = _mm256_permute2f128_pd(row01lo, row01hi, 0x20);
//             __m256d comb2row01hi = _mm256_permute2f128_pd(row01lo, row01hi, 0x31);

//             __m256d row23lo = _mm256_unpacklo_pd(row2, row3);
//             __m256d row23hi = _mm256_unpackhi_pd(row2, row3);
            
//             __m256d comb2row23lo = _mm256_permute2f128_pd(row23lo, row23hi, 0x20);
//             __m256d comb2row23hi = _mm256_permute2f128_pd(row23lo, row23hi, 0x31);

//             __m256d row45lo = _mm256_unpacklo_pd(row4, row5);
//             __m256d row45hi = _mm256_unpackhi_pd(row4, row5);
            
//             __m256d comb2row45lo = _mm256_permute2f128_pd(row45lo, row45hi, 0x20);
//             __m256d comb2row45hi = _mm256_permute2f128_pd(row45lo, row45hi, 0x31);

//             __m256d comb4row0123c1 = _mm256_permute2f128_pd(comb2row01lo, comb2row23lo, 0x20);
//             __m256d comb4row0123c2 = _mm256_permute2f128_pd(comb2row01lo, comb2row23lo, 0x31);
//             __m256d comb4row0123c3 = _mm256_permute2f128_pd(comb2row01hi, comb2row23hi, 0x20);
//             __m256d comb4row0123c4 = _mm256_permute2f128_pd(comb2row01hi, comb2row23hi, 0x31);


//             __m256d comb42c2hi = _mm256_permute2f128_pd(comb2row45lo, comb4row0123c2, 0x20);
//             __m256d comb42c2lo = _mm256_permute2f128_pd(comb4row0123c2, comb2row45lo, 0x31);
//             __m256d comb42c4hi = _mm256_permute2f128_pd(comb2row45hi, comb4row0123c4, 0x20);
//             __m256d comb42c4lo = _mm256_permute2f128_pd(comb4row0123c4, comb2row45hi, 0x31);
            
//             _mm256_store_pd(&pack_a[block * 8 * z + col * 6], comb4row0123c1);
//             _mm256_store_pd(&pack_a[block * 8 * z + col * 6 + 4], comb42c2hi);
//             _mm256_store_pd(&pack_a[block * 8 * z + col * 6 + 8], comb42c2lo);
//             _mm256_store_pd(&pack_a[block * 8 * z + col * 6 + 12], comb4row0123c3);
//             _mm256_store_pd(&pack_a[block * 8 * z + col * 6 + 16], comb42c4hi);
//             _mm256_store_pd(&pack_a[block * 8 * z + col * 6 + 20], comb42c4lo);
//         }

//     }
// }



// The example matrix is 2x4x8 (height x width x channels), the input matrix is row major ordering

// 0 1 2 3 4 5 6 7
// 8 9 10 11 12 13 14 15
// 16 17 18 19 20 21 22 23
// 24 25 26 27 28 29 30 31
// 32 33 34 35 36 37 38 39
// 40 41 42 43 44 45 46 47
// 48 49 50 51 52 53 54 55
// 56 57 58 59 60 61 62 63

// First route compates by pairing together the first two rows, then the next two rows, and so on
// 0 8 1 9 2 10 3 11 4 12 5 13 6 14 7 15
// 16 24 17 25 18 26 19 27 20 28 21 29 22 30 23 31
// 32 40 33 41 34 42 35 43 36 44 37 45 38 46 39 47
// 48 56 49 57 50 58 51 59 52 60 53 61 54 62 55 63

// Second route compares by pairing together the first and third rows, then the second and fourth rows, and so on
// 0 8 16 24 1 9 17 25 2 10 18 26 3 11 19 27 4 12 20 28 5 13 21 29 6 14 22 30 7 15 23 31
// 32 40 48 56 33 41 49 57 34 42 50 58 35 43 51 59 36 44 52 60 37 45 53 61 38 46 54 62 39 47 55 63

// The final result is the concatenation of the two routes
// 0 8 16 24 32 40 48 56 1 9 17 25 33 41 49 57 2 10 18 26 34 42 50 58 3 11 19 27 35 43 51 59 4 12 20 28 36 44 52 60 5 13 21 29 37 45 53 61 6 14 22 30 38 46 54 62 7 15 23 31 39 47 55 63

// Since we do not have to modify the internal order of the matrix, we are just switching the ordering for every K layers



// 0 1 2 3 .....  66 67 68
// 69 70 71 72 ... 135 136 137
// 138 139 140 141 ... 204 205 206