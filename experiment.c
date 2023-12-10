#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <mpi.h>

#include "pooling.h"
#include "pack.h"
#include "utils.h"

#define RUNS 1000
#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

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
    int print_processor = 0; //who prints their data.

    float *source, *pack_source;
    float *output, *output_check, *pack_output;

    unsigned long long bt0, bt1, ot0, ot1;
    double oavg, bavg;
    int correct = 1;

    int m = 4;      // m is the number of rows per layer 
    int n = 4;      // n is the number of columns per layer 
    
    int pool = 2;   // pool is the size of the pooling window

    int z = m * n;  // z is the size of a layer (m * n)


    /*
    Assume the following
        - All matrices are storedin row major order
    */
    if (world_rank == print_processor) printf("p\t z\t k\t oflops\t bflops\t correct\n");
    for (int k = 32; k <= 512; k+= 32){    // k is the number of layers

        posix_memalign((void**) &source, 64, z * k * sizeof(float));
        posix_memalign((void**) &pack_source, 64, z * k * sizeof(float));
        posix_memalign((void**) &output, 64, m * n / pool / pool * k * sizeof(float));
        posix_memalign((void**) &pack_output, 64, m * n / pool / pool * k * sizeof(float));
        posix_memalign((void**) &output_check, 64, m * n / pool / pool * k * sizeof(float));

        oavg = 0;
        bavg = 0;

        for (int r = 0; r != RUNS; ++r){

            //initialize input matrix 
            for (int i = 0; i != k * z; ++i){
                source[i] = ((float) rand())/ ((float) RAND_MAX);
                pack_source[i] = 0;
            }

            pack(pack_source, source, m, n, k, pool);
            ot0 = rdtsc();
            kernel(pack_output, pack_source, k, m, n, pool);
            ot1 = rdtsc();
            unpack(output, pack_output, m*n/pool/pool, k);
            double ocycles = (double)(ot1 - ot0);
            // printf("oflops: %lf\n", (m*n*k)/(ocycles*MAX_FREQ/BASE_FREQ));
            oavg += (1.0*m*n*k)/(ocycles*MAX_FREQ/BASE_FREQ);

            bt0 = rdtsc();
            naive(output_check, source, k, m, n, pool);
            bt1 = rdtsc();
            double bcycles = (double)(bt1 - bt0);
            bavg += (m*n*k)/(bcycles*MAX_FREQ/BASE_FREQ);

            correct &= compare_matrix(output, output_check, k*m*n/pool/pool);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        printf("%d\t %d\t %d\t %3.3lf\t %3.3lf\t %d\n", world_rank, z, k, oavg/(RUNS*1.0), bavg/(RUNS*1.0), correct);
    
    }

    free(source);
    free(pack_source);
    free(output);
    free(pack_output);
    free(output_check);

    MPI_Finalize();

    return 0;
}