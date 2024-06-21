#ifndef VECTOR_ADDITION
#define VECTOR_ADDITION



__global__ void vecAddKernel(float* A, float* B, float* C, int n);

void vecAdd(float* A_h, float* B_h, float* C_h, int n);



#endif