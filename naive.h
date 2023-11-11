/*
    Baseline implementation of the pooling layer
*/



void naive
(
    float*      restrict output,    // pointer to the output matrix
    float*      restrict input,     // pointer to the input matrix
    int         k,                  // number of layers
    int         m,                  // number of rows per layer
    int         n,                  // number of columns per layer
    int         pool                // pooling size
) {

    int m_out = m / pool;
    int n_out = n / pool;

    for (int d = 0; d < k; d++) { // for each layer

        for (int i = 0; i < m_out; i++) {

            for (int j = 0; j < n_out; j++) {

                float max_val = input[d * m * n + i * pool * n + j * pool];

                for (int x = 0; x < pool; x++) {

                    for (int y = 0; y < pool; y++) {

                        float current_val = input[d * m * n + (i * pool + x) * n + j * pool + y];

                        if (current_val > max_val) {
                            max_val = current_val;
                        }

                    }

                }

                output[d * m_out * n_out + i * n_out + j] = max_val;
            }
        }
    }
}
