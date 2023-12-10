#ifndef PACK_H
#define PACK_H

void pack // naive packing routine
(
    float*     restrict pack_source, // result data
    float*     restrict source,      // source data
    int               m,             // number of rows per layer
    int               n,             // number of columns per layer
    int               k,              // number of layers
    int              pool           // size of the pooling window
);

void unpack ( // naive packing routine
    float*     restrict res,              // result data
    float*     restrict pack_res,         // source data in packed format
    int               z,                  // size of a layer (m * n)
    int               k                   // number of layers
);

#endif
