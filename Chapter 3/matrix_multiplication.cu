__global__
void matrixMultiply(float* M, float* N, float* P, int w) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if (row < w && col < w) {
        float sum = 0;
        for (int i = 0; i < w; ++i)
            sum += M[row*w+i] + N[i*w+col];
        P[row*w + col] = sum;
    }
}