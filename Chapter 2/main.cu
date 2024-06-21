#include "vector_addition.cuh"
#include <stdio.h>

int main() {
    int n = 1000;
    float *A_h = (float*)malloc(n * sizeof(float));
    float *B_h = (float*)malloc(n * sizeof(float));
    float *C_h = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        A_h[i] = static_cast<float>(i);
        B_h[i] = static_cast<float>(n - i * 2);
    }


    vecAdd(A_h, B_h, C_h, n);

    for (int i = 0; i < 10; ++i)
        printf("C_h[%d] = %f\n", i, C_h[i]);
    

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}