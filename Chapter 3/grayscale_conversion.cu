__global__
void colortoGrayscaleConversion(unsigned char* Pout, 
                unsigned char* Pin, int width, int height) {
    // By using blockIdx.x and threadIdx.x for columns, and blockIdx.y and threadIdx.y for rows, 
    // CUDA ensures that threads with consecutive thread IDs access consecutive memory locations. This can lead to better memory access patterns and improved performance.
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row*width + col;
        // One can think of the RGB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset*CHANNELS
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}
