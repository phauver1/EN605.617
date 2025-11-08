#include "boids.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <vector>
#include <cmath>

// Wrapper for API calls
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err_), err_); \
        throw std::runtime_error(cudaGetErrorString(err_)); \
    } \
} while(0)

typedef unsigned int u32;

// Make vector helper functions
__device__ inline float2 add(const float2& a, const float2& b) {
    return make_float2(a.x+b.x, a.y+b.y); // Add two vectors
}
__device__ inline float2 sub(const float2& a, const float2& b) {
    return make_float2(a.x-b.x, a.y-b.y); // Subtract one vector from another
}
__device__ inline float2 scale(const float2& a, float s) {
    return make_float2(a.x*s, a.y*s); // Scale a vector from an hour
}
__device__ inline float length(const float2& a) {
    return sqrtf(a.x*a.x + a.y*a.y); // Take the magnitude of a vector
}
__device__ inline float2 normalize(const float2& a) {
    float m=length(a)+1e-6f; return make_float2(a.x/m,a.y/m); // normalize a vector
}

// --- RNG init ---
__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* rng, 
                                std::uint64_t seed, std::uint32_t numBoids) {
    std::uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (u32 i=tid; i<numBoids; i+=stride) { // Loop until we've reached the end of the array
        if (i < numBoids) curand_init(seed, i, 0, &rng[i]);
    }
}

// Initialize the boids
__global__ void init_boids_kernel(float* posx,float* posy,float* velx,float* vely,
                                  std::uint32_t numBoids,BoidsParams params,
                                  curandStatePhilox4_32_10_t* rng){
    std::uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x; // Get the thread index
	u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
	for (u32 i=tid; i<numBoids; i+=stride) { // Loop until we've reached the end of the array
        // Generate a random position for each boid
        curandStatePhilox4_32_10_t local = rng[i];
        float rx = curand_uniform(&local);
        float ry = curand_uniform(&local);
        posx[i]=rx*params.world_width;
        posy[i]=ry*params.world_height;
        // Generate a random velocity for each boid
        float theta = curand_uniform(&local) * 6.2831853f;
        float2 v = make_float2(cosf(theta), sinf(theta));
        v = scale(normalize(v), 0.5f*params.max_speed);
        velx[i]=v.x; vely[i]=v.y;
        rng[i] = local; // Stores the advanced random draw back into memory
    }
}

// Step the boids forward in time
__global__ void boids_step_kernel(const float* posx,const float* posy,
    float* velx,float* vely,
    float* out_posx,float* out_posy,
    std::uint32_t numBoids,BoidsParams params,float dt){

    std::uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x; // Get the thread index
    u32 stride = blockDim.x * gridDim.x; // Step over the number of threads in each block/grid
    for (u32 i=tid; i<numBoids; i+=stride) { // Loop until we've reached the end of the array

        // Record the position of the boid
        float px=posx[i], py=posy[i];

        // Record the velocity of the boid in vector form
        float2 v=make_float2(velx[i], vely[i]);
        float2 vn = normalize(v);

        // Record the number of neighbor influences we've had
        int num_neighbors = 0;

        // Create vectors for alignment, cohesion, and acceleration.
        float2 align_acc=make_float2(0,0), cohese_acc=make_float2(0,0), separate_acc=make_float2(0,0);

        // Iterate through each nearby boid to determine weights
        for(std::uint32_t j=0;j<numBoids;++j){
            if(j==i) continue; // Don't double-dip with the current boid
            float dx=posx[j]-px+1e-6f, dy=posy[j]-py+1e-6f;
            float d = sqrt(dx*dx+dy*dy);
            if(d>params.perception_radius) continue;
            // Calculate the off-angle from the ownship
            float cos_theta = (vn.x*dx + vn.y*dy)/length(v);
            // If the off-angle is too high, skip
            if (cos_theta < params.angle_limit) continue;
            // Iterate the number of neighbors
            num_neighbors ++;
            // Align the boid with its neighbors
            align_acc=add(align_acc,make_float2(velx[j],vely[j]));
            // Keep the boid close to its neighbors
            cohese_acc=add(cohese_acc,make_float2(posx[j],posy[j]));
            // Keep the boid away from its neighbors
            separate_acc=add(separate_acc,scale(make_float2(-dx,-dy),1.f/d));
        }

        // Steering composition
        float2 steer=make_float2(0,0);
        if (num_neighbors) {
            // Scale the alignment value and add to the steering force
            align_acc=scale(align_acc, 1.0f/num_neighbors);
            steer = add(steer,scale(align_acc,params.align_weight/8.f));
            // Scale the cohesion value and add to the steering force
            cohese_acc = sub(scale(cohese_acc, 1.0f/num_neighbors), make_float2(px, py));
            steer = add(steer,scale(cohese_acc,params.cohese_weight));
            // Add the separation value to the steering force
            steer = add(steer,scale(separate_acc,params.separate_weight*2.f));
        }

        // Boundary separation (repel when within perception radius of walls)
        {
            float2 boundary_acc = make_float2(0,0);
            float r = 20; // params.perception_radius;

            // Get the distance from the left boundary
            float px_inv = params.world_width  - px;
            // Get the distance from the top boundary
            float py_inv = params.world_height - py;
            
            // Apply a weight opposite from each boundary if within a certain distance
            if (px     < r) boundary_acc = add(boundary_acc, make_float2((r-px)/(r), 0.0f));
            if (px_inv < r) boundary_acc = add(boundary_acc, make_float2((px_inv-r)/(r), 0.0f));
            if (py     < r) boundary_acc = add(boundary_acc, make_float2(0.f, (r-py)/(r)));
            if (py_inv < r) boundary_acc = add(boundary_acc, make_float2(0.f,(py_inv-r)/(r)));

            // Apply the boudnary force to the steering vector
            if (boundary_acc.x != 0.0f || boundary_acc.y != 0.0f){
                steer = add(steer, scale(boundary_acc, params.boundary_weight));
            }
        }

        // Apply effect of various forces to velocity
        float scaleFactor = params.max_speed/(length(v)+1e-6f);
        v = add(scale(v,params.hysteresis*scaleFactor),scale(steer,(1.-params.hysteresis)*scaleFactor));
        
        // Propogate the position of the boid acccording to the velocity
        px+=v.x*dt; py+=v.y*dt;

        // Reflect the velocity across the boundary if we accidentally cross
        if(px<0){px=0; v.x=fabsf(v.x);} // Left boundary
        if(px>params.world_width){px=params.world_width; v.x=-fabsf(v.x);} // Right boundary
        if(py<0){py=0; v.y=fabsf(v.y);} // Bottom boundary
        if(py>params.world_height){py=params.world_height; v.y=-fabsf(v.y);} // Top boundary
        
        // Save the velocity
        velx[i]=v.x; vely[i]=v.y;
        // Save the position
        out_posx[i]=px; out_posy[i]=py;
    }
}


// Constructor
BoidsSim::BoidsSim(std::uint32_t num_boids,BoidsParams params,std::uint64_t seed)
 :N_(num_boids),params_(params),d_posx_(nullptr),d_posy_(nullptr),d_velx_(nullptr),d_vely_(nullptr),d_rng_(nullptr){
    allocate(N_);
    dim3 block(256), grid((N_+block.x-1)/block.x);
    CUDA_CHECK(cudaMalloc((void**)&d_rng_, N_*sizeof(curandStatePhilox4_32_10_t)));
    init_rng_kernel<<<grid,block>>>((curandStatePhilox4_32_10_t*)d_rng_, seed, N_);
    CUDA_CHECK(cudaDeviceSynchronize());
    init_boids_kernel<<<grid,block>>>(d_posx_,d_posy_,d_velx_,d_vely_,N_,params_,(curandStatePhilox4_32_10_t*)d_rng_);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Destructor
BoidsSim::~BoidsSim(){ free(); }

// Allocate boid state variables
void BoidsSim::allocate(std::uint32_t n){
    CUDA_CHECK(cudaMalloc((void**)&d_posx_, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_posy_, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_velx_, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_vely_, n*sizeof(float)));
}

// Deallocate boid state varibles
void BoidsSim::free(){
    if(d_posx_) cudaFree(d_posx_); d_posx_=nullptr;
    if(d_posy_) cudaFree(d_posy_); d_posy_=nullptr;
    if(d_velx_) cudaFree(d_velx_); d_velx_=nullptr;
    if(d_vely_) cudaFree(d_vely_); d_vely_=nullptr;
    if(d_rng_)  cudaFree(d_rng_); d_rng_=nullptr;
}

// Changes the current number of boids in the sim
void BoidsSim::resize(std::uint32_t num_boids){
    if (num_boids == N_) return;
    free();
    N_ = num_boids;
    allocate(N_);

    dim3 block(256), grid((N_ + block.x - 1) / block.x);
    CUDA_CHECK(cudaMalloc((void**)&d_rng_, N_ * sizeof(curandStatePhilox4_32_10_t)));

    init_rng_kernel<<<grid, block>>>(
        (curandStatePhilox4_32_10_t*)d_rng_, 1234ULL, N_
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    init_boids_kernel<<<grid, block>>>(
        d_posx_, d_posy_, d_velx_, d_vely_,
        N_, params_,
        (curandStatePhilox4_32_10_t*)d_rng_
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Set simulation parameters
void BoidsSim::set_params(const BoidsParams& p){ params_ = p; }

// Return simulation parameters
BoidsParams BoidsSim::get_params() const { return params_; }

// Return number of boids
std::uint32_t BoidsSim::count() const {return N_; }

// Steps the simulation to the next frame
void BoidsSim::step(float dt){
    dim3 block(256), grid((N_ + block.x - 1) / block.x);
    boids_step_kernel<<<grid, block>>>(
        d_posx_, d_posy_,
        d_velx_, d_vely_,
        d_posx_, d_posy_,
        N_, params_, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copies all boid positions from the GPU to CPU
std::vector<float> BoidsSim::positions_host() const {
    std::vector<float> out(2 * N_);
    CUDA_CHECK(cudaMemcpy(out.data(), d_posx_, N_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data() + N_, d_posy_, N_ * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

// Copies all boid velocities from the GPU to the CPU
std::vector<float> BoidsSim::velocities_host() const {
    std::vector<float> out(2 * N_);
    CUDA_CHECK(cudaMemcpy(out.data(), d_velx_, N_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data() + N_, d_vely_, N_ * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}
