//Based on the work of Andrew Krepps
#include <stdio.h>
#include <chrono> // Included so we can track start/stop time

typedef unsigned int u32;
#define ARRAY_SIZE 1 << 12 // Set a large array size: 2^20 aka 1,048,576
#define DATA_ARRAY_SIZE_IN_BYTES (sizeof(u32) * (ARRAY_SIZE)) // Size of this array in unsigned ints
__constant__ u32 const_data[ARRAY_SIZE];

// Declare pointers for host based params
u32* host_data = (u32*)malloc(DATA_ARRAY_SIZE_IN_BYTES);

// Declare pointers for GPU based params
u32 *global_data;

void SquareArrayHost(u32* data, u32 len_data) {
	// Square the values in an array using purely host memory
	for (int i=0; i<len_data; i++) {
		u32 value = data[i]; // Read data into host memory
		data[i] *= value; // Square the value in the data array
	}
}

__global__ void SquareArraySharedViaGlobal(u32* data, u32 const len_data) {
	// Square the values in an array using shared memory
	extern __shared__ u32 shared_data[];
	u32 tid = threadIdx.x + blockIdx.x * blockDim.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=tid; i<len_data; i+=stride) { // Loop until we've reached the end of the array
		shared_data[threadIdx.x] = data[i]; // Read data into shared memory
		// __syncthreads(); // Wait for all the threads to reach this step
		data[i] *= shared_data[threadIdx.x]; // Square the value in the data array
	}
}

__global__ void SquareArrayRegisterViaGlobal(u32* data, u32 const len_data) {
	// Square the values in an array using register memory
	u32 tid = threadIdx.x + blockIdx.x * blockDim.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=tid; i<len_data; i+=stride) { // Loop until we've reached the end of the array
		register u32 value = data[i]; // Read data into register memory
		data[i] *= value; // Square the value in the data array
	}
}

__global__ void SquareArraySharedViaConst(u32* data, u32 const len_data) {
	// Square the values in an array using shared memory
	extern __shared__ u32 shared_data[];
	u32 tid = threadIdx.x + blockIdx.x * blockDim.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=tid; i<len_data; i+=stride) { // Loop until we've reached the end of the array
		shared_data[threadIdx.x] = const_data[i]; // Read data into shared memory
		// __syncthreads(); // Wait for all the threads to reach this step
		data[i] *= shared_data[threadIdx.x]; // Square the value in the data array
	}
}

__global__ void SquareArrayRegisterViaConst(u32* data, u32 const len_data) {
	// Square the values in an array using register memory
	u32 tid = threadIdx.x + blockIdx.x * blockDim.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=tid; i<len_data; i+=stride) { // Loop until we've reached the end of the array
		register u32 value = const_data[i]; // Read data into register memory
		data[i] *= value; // Square the value in the data array
	}
}

void RunAndTime(u32 numBlocks, u32 threads_per_block, u32 blockSize,
		u32* host_data, u32* global_data) {
	// Run and time the Host function
    auto start = std::chrono::high_resolution_clock::now(); // Record start time
	SquareArrayHost(host_data, ARRAY_SIZE);
	auto stop = std::chrono::high_resolution_clock::now(); // Record finish time
	double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("Host time: %f ms\n", elapsed_time/1000.); // Print the results

	// Run and time the SharedViaGlobal GPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	SquareArraySharedViaGlobal<<<numBlocks, threads_per_block, blockSize*sizeof(u32)>>>(global_data, ARRAY_SIZE);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("Shared/Global time: %f ms\n", elapsed_time/1000.); // Print the results

	// Run and time the RegisterViaGlobal GPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	SquareArrayRegisterViaGlobal<<<numBlocks, threads_per_block>>>(global_data, ARRAY_SIZE);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("Register/Global time: %f ms\n", elapsed_time/1000.); // Print the results

	// Run and time the SharedViaConst GPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	SquareArraySharedViaConst<<<numBlocks, threads_per_block, blockSize*sizeof(u32)>>>(global_data, ARRAY_SIZE);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("Shared/Cont time: %f ms\n", elapsed_time/1000.); // Print the results

	// Run and time the RegisterViaConst GPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	SquareArrayRegisterViaConst<<<numBlocks, threads_per_block>>>(global_data, ARRAY_SIZE);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("Register/Const time: %f ms\n", elapsed_time/1000.); // Print the results
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) totalThreads = atoi(argv[1]);
	if (argc >= 3) blockSize = atoi(argv[2]);

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Calculate how many thread that shakes out to
	int threads_per_block = totalThreads/numBlocks;

	// Set the value of the value array
	for (u32 i = 0; i < ARRAY_SIZE; ++i) host_data[i] = i;

	// Allocate memory for the global array
	cudaMalloc((void **)&global_data, DATA_ARRAY_SIZE_IN_BYTES);

	// Copy value_array to the GPU
	cudaMemcpy(global_data, host_data, DATA_ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(const_data, host_data, DATA_ARRAY_SIZE_IN_BYTES);

	// Run and time the assigned functions
	RunAndTime(numBlocks, threads_per_block, blockSize, host_data, global_data);

	// Free the global array on the GPU
	cudaFree(global_data);
}
