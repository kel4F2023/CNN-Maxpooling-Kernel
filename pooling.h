#ifndef POOLING_H
#define POOLING_H

void kernel 
(
    float*      restrict output,    // pointer to the output matrix
    float*      restrict input,     // pointer to the input matrix
    int         k,                  // number of layers
    int         m,                  // number of rows per layer
    int         n,                  // number of columns per layer
    int         pool                // pooling size
);

void naive
(
    float*      restrict output,    // pointer to the output matrix
    float*      restrict input,     // pointer to the input matrix
    int         k,                  // number of layers
    int         m,                  // number of rows per layer
    int         n,                  // number of columns per layer
    int         pool                // pooling size
);

#endif