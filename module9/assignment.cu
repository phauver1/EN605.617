#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <math.h>
#include <time.h>


#define nStocks 25 // Number of stocks to analyze
#define blockSize 256   // length of each stock's time series
#define nDays 365   // length of each stock's time series

__global__ void init(unsigned int seed, curandState_t* states,
        const unsigned int size) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=thread_idx; i<size; i+=stride) { // Loop until we've reached the end of the array
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void randoms(curandState_t* states, float* numbers,
        const unsigned int size) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=thread_idx; i<size; i+=stride) { // Loop until we've reached the end of the array
        // Generate the random number
        float rand_number = curand_normal(&states[i]);
        // Cap the number to -5,5 and return
        numbers[i] = fminf(fmaxf(rand_number, -5.0f), 5.0f);
    }
}

int main() {
    curandState_t* states; // Allocate space for the cuRAND states
    cudaMalloc((void**) &states, nStocks * nDays * sizeof(curandState_t));
    init<<<nStocks, blockSize>>>(time(0), states, nStocks*nDays); // Initialize the random states
    float* gpu_nums; // Allocate an array of floats on the GPU
    cudaMalloc((void**) &gpu_nums, nStocks * nDays * sizeof(float));
    randoms<<<nStocks, blockSize>>>(states, gpu_nums, nStocks*nDays); // Generate random numbers

    cufftHandle plan; // Setup cuFFT plan
    cufftPlan1d(&plan, nDays, CUFFT_R2C, nStocks); // Batch 1D FFT for each stock
    cufftComplex* gpu_freq; // Allocate an array for FFT results
    cudaMalloc((void**)&gpu_freq, nStocks*(nDays/2+1)*sizeof(cufftComplex));
    cufftExecR2C(plan, gpu_nums, gpu_freq); // Execute FFT
    
    // Copy the results back to the CPU
    cufftComplex* cpu_freq = new cufftComplex[nStocks*(nDays/2+1)];
    cudaMemcpy(cpu_freq, gpu_freq, nStocks*(nDays/2+1)*sizeof(cufftComplex),
        cudaMemcpyDeviceToHost);
    
    // Print spectrum magnitude for each stock
    for (int i = 0; i < nStocks; i++) {
        printf("Stock %d FFT magnitudes:\n", i);
        for (int j = 0; j < nDays/2 + 1; j++) {
            float re = cpu_freq[i*(nDays/2+1) + j].x;
            float im = cpu_freq[i*(nDays/2+1) + j].y;
            float mag = sqrtf(re*re + im*im)/nDays;
            printf("%f ", mag);
        }
        printf("\n\n");
    }
    
    delete[] cpu_freq; // Delete the CPU copy of the FFT results
    cufftDestroy(plan); // Delete the cuFFT plan
    cudaFree(states); // Delete the random states
    cudaFree(gpu_nums); // Delete the cuRAND generated random numbers
    cudaFree(gpu_freq); // Delete the GPU copy of the FFT results
    return 0; // End the mainloop
}
