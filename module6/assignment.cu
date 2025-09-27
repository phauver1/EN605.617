//Based on the work of Andrew Krepps
#include <stdio.h>

typedef unsigned int u32;
#define ARRAY_SIZE (1 << 20) // Set a large array size: 2^20 aka 1,048,576
#define DATA_ARRAY_SIZE_IN_BYTES (sizeof(float) * (ARRAY_SIZE)) // Size of this array in unsigned ints

__global__
void saxpy(int len_data, float a, float *x, float *y)
{
	u32 tid = blockIdx.x*blockDim.x + threadIdx.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (u32 i=tid; i<len_data; i+=stride) { // Loop until we've reached the end of the array
		y[i] = a*x[i] + y[i];
	}
}

__global__ void reverseArray(u32 len_data, float *array)
{
	u32 tid = blockIdx.x*blockDim.x + threadIdx.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (u32 i=tid; i<len_data; i+=stride) { // Loop until we've reached the end of the array
	u32 i_r = len_data-i-1; // Get the reverse of the thread index
	if (i > i_r) break;
		float value = array[i]; // Save the value of the array entry
		array[i] = array[i_r]; // Replace the array entry with the reverse index of the array
		array[i_r] = value; // Replace the array reverse entry with the array entry
	}
}

void runStreamAndTime(u32 numBlocks, u32 blockSize, float*host_x, float*host_y, float*gpu_x, float*gpu_y) {
	// Run the stream assignment and time the results

	// Initialize the variables needed for timing
	float elapsedTime;
	cudaEvent_t start, stop;

	// Create start and stop events
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 

	// Create a stream
	cudaStream_t stream; 
	cudaStreamCreate(&stream);

	// Initialize the host arrays
	for(int index = 0; index < ARRAY_SIZE; index++) 
	{ 
		host_x[index] = (float)(rand()%10000)/100.f; 
		host_y[index] = (float)(rand()%10000)/100.f; 
	}

	cudaMemcpyAsync(gpu_x, host_x,DATA_ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice, stream); // Copy the host arrays to the gpus
	cudaMemcpyAsync(gpu_y, host_y, DATA_ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice, stream); // Copy the host arrays to the gpus
	cudaEventRecord(start); // start recording the time
	saxpy <<<numBlocks, blockSize, 0, stream>>>(ARRAY_SIZE, 3.0, gpu_x, gpu_y); // Run the saxpy calc on the two arrays
	cudaStreamSynchronize(stream); // Wait for the all the 'saxpy' commands to finish
	reverseArray <<<numBlocks, blockSize, 0, stream>>>(ARRAY_SIZE, gpu_y); // Reverse the result from the saxpy calc
	cudaStreamSynchronize(stream); // Wait for the reverse command to finish
	cudaMemcpyAsync(host_y, gpu_y, DATA_ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost, stream); // Copy the result to the host
	cudaStreamSynchronize(stream); // Wait for all commands to finish

	cudaEventRecord(stop, 0); // Record the stop time
	cudaEventSynchronize(stop); // Wait until the 'stop' event has been reached
	cudaEventElapsedTime(&elapsedTime, start, stop); // Compute the ellapsed time

	printf("Number of blocks : %d, Block Size : %d\n", numBlocks, blockSize);
	printf("\tTime taken: %f ms \n", elapsedTime); 
}



int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Initialize the host and gpu arrays
	float *host_x, *host_y, *gpu_x, *gpu_y;

	// Allocate the gpu arrays in gpu shared memory
	cudaMalloc( ( void**)& gpu_x, DATA_ARRAY_SIZE_IN_BYTES); 
	cudaMalloc( ( void**)& gpu_y, DATA_ARRAY_SIZE_IN_BYTES);
  
	// Allocate pinned memory for the host arrays
	cudaHostAlloc((void **)&host_x, DATA_ARRAY_SIZE_IN_BYTES, cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_y, DATA_ARRAY_SIZE_IN_BYTES, cudaHostAllocDefault);

	// Run the assignment code
	runStreamAndTime(numBlocks, blockSize, host_x, host_y, gpu_x, gpu_y);

	// De-allocate the host and gpu arrays
	cudaFreeHost(host_x);
	cudaFreeHost(host_y);
	cudaFree(gpu_x);
	cudaFree(gpu_y);

	return 0; 
}
