#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <mpi.h>

#include "pooling.h"
#include "pack.h"
#include "utils.h"

#define RUNS 100000
#define MAX_FREQ 2.9
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

    float *source, *output;

    unsigned long long ot0, ot1;
    double oavg;

    int m = 4;      // m is the number of rows per layer 
    int n = 4;      // n is the number of columns per layer 
    int k = 32;     // k is the number of layers (multiple of 32)
    if (argc > 1) k = atoi(argv[1]);
    int pool = 2;   // pool is the size of the pooling window
    int z = m * n;  // z is the size of a layer (m * n)

    if (world_rank == print_processor) printf("p\t z\t k\t oflops\t\n");

    posix_memalign((void**) &source, 64, z * k * sizeof(float));
    posix_memalign((void**) &output, 64, m * n / pool / pool * k * sizeof(float));

    oavg = 0;

    for (int r = 0; r != RUNS; ++r){

        //initialize input matrix 
        for (int i = 0; i != k * z; ++i){
            source[i] = 1;
        }
        ot0 = rdtsc();
        kernel(output, source, k, m, n, pool);
        ot1 = rdtsc();
        double ocycles = (double)(ot1 - ot0);
        oavg += (1.0*m*n*k)/(ocycles*MAX_FREQ/BASE_FREQ);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("%d\t %d\t %d\t %3.3lf\t\n", world_rank, z, k, oavg/(RUNS*1.0));
    
    

    free(source);
    free(output);

    MPI_Finalize();

    return 0;
}