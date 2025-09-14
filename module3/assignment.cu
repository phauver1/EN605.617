//Based on the work of Andrew Krepps
#include <stdio.h> // Included (presumably) so we can use the printf function
#include <chrono> // Included so we can track start/stop time

#define ARRAY_SIZE 1 << 20 // Set a large array size: 2^20 aka 1,048,576
#define VALUE_ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE)) // Size of this array in unsigned ints
#define RESULT_ARRAY_SIZE_IN_BYTES (sizeof(bool) * (ARRAY_SIZE)) // Size of this array in bools

// Define the 'branching' CUDA funtion
__global__ // Keyword makes this function runnable from the GPU
void CUDA_divisible_by_two_branching(unsigned int * value_array, bool * result_array, const int size)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; // Get the thread index
	int stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=thread_idx; i<size; i+=stride) { // Loop until we've reached the end of the array
		if (value_array[i]%2==0) {
			result_array[i] = true; // The number is divisible by two
		}
		else {
			result_array[i] = false; // The number is not divisible by two
		}
	}
}

// Define the 'non-branching' CUDA funtion
__global__ // Keyword makes this function runnable from the GPU
void CUDA_divisible_by_two_nonbranching(unsigned int * value_array, bool * result_array, const int size)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; // Get the thread index
	int stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (int i=thread_idx; i<size; i+=stride) { // Loop until we've reached the end of the array
		result_array[i] = (value_array[i]%2==0); // Record whether the number was divisible by two directly
	}
}

// Define the 'branching' CPU funtion
void CPU_divisible_by_two_branching(unsigned int * value_array, bool * result_array, const int size)
{
	for (unsigned int i=0; i<size; i++) { // Iterate through each of the values in the array
		if (value_array[i]%2==0) {
			result_array[i] = true; // The number is divisible by two
		}
		else {
			result_array[i] = false; // The number is not divisible by two
		}
	}
}

// Define the 'non-branching' CPU funtion
void CPU_divisible_by_two_nonbranching(unsigned int * value_array, bool * result_array, const int size)
{
	for (unsigned int i=0; i<size; i++) { // Iterate through each of the values in the array
		result_array[i] = (value_array[i]%2); // Record whether the number was divisible by two directly
	}
}

int main(int argc, char** argv)
{
	// Set defaults for the command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	// Read any command line arguments
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	// Calculate the number of blocks we need based on the inputs, and how many thread that shakes out to
	int numBlocks = totalThreads/blockSize;
	int threads_per_block = totalThreads/numBlocks;

	// If the total number of threads is not a multiple of the blocksize, throw a warning.
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Initialize the common variables we will use
	std::chrono::time_point<std::chrono::high_resolution_clock> start{}; 
	std::chrono::time_point<std::chrono::high_resolution_clock> stop{};
	double elapsed_time{0};
	unsigned int value_array[ARRAY_SIZE]; // This is the array of values we are testing
	bool result_array[ARRAY_SIZE]; // This is where we store the results of the test

	// Set the value of the value array
	for (unsigned int i = 0; i < ARRAY_SIZE; ++i) {
		value_array[i] = i;
	}

	// Declare pointers for GPU based params
	unsigned int *gpu_value_array;
	bool *gpu_result_array;

	// Allocate memory for our arrays
	cudaMalloc((void **)&gpu_value_array, VALUE_ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_result_array, RESULT_ARRAY_SIZE_IN_BYTES);

	// Copy value_array to the GPU
	cudaMemcpy(gpu_value_array, value_array, VALUE_ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	// Run and time the branching GPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	CUDA_divisible_by_two_branching<<<numBlocks, threads_per_block>>>(gpu_value_array, gpu_result_array, ARRAY_SIZE);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("GPU branching time: %f ms\n", elapsed_time/1000.); // Print the results

    // Run and time the non-branching GPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	CUDA_divisible_by_two_nonbranching<<<numBlocks, threads_per_block>>>(gpu_value_array, gpu_result_array, ARRAY_SIZE);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("GPU non-branching time: %f ms\n", elapsed_time/1000.); // Print the results

	// Free the arrays on the GPU as now we're done with them
	cudaFree(gpu_value_array);
	cudaFree(gpu_result_array);

	// Run and time the branching CPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	CPU_divisible_by_two_branching(value_array, result_array, ARRAY_SIZE);
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("CPU branching time: %f ms\n", elapsed_time/1000.); // Print the results

	// Run and time the non-branching CPU function
    start = std::chrono::high_resolution_clock::now(); // Record start time
	CPU_divisible_by_two_nonbranching(value_array, result_array, ARRAY_SIZE);
	stop = std::chrono::high_resolution_clock::now(); // Record finish time
	elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
	printf("CPU non-branching time: %f ms\n", elapsed_time/1000.); // Print the results
}
