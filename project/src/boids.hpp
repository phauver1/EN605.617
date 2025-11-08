/*
    Header file for boids.cu
*/

#pragma once
#include <cstdint>
#include <vector>

// Simulation parameters
struct BoidsParams {
    float perception_radius; // neighborhood radius for all three rules
    float angle_limit;       // Cosine of half-angle of perception
    float max_speed;         // maximum velocity magnitude
    float hysteresis;        // weight for current velocity
    float align_weight;      // weight for alignment (-10 to +10)
    float cohese_weight;     // weight for cohesion (-10 to +10)
    float separate_weight;   // weight for separation (-10 to +10)
    float boundary_weight;   // strength of repulsion from nearest boundary
    float world_width;       // simulation world width
    float world_height;      // simulation world height
};

// Define boids simulation class
class BoidsSim {
public:
    // Constructor
    BoidsSim(std::uint32_t num_boids, BoidsParams params, std::uint64_t seed = 1234);
    // Destructor
    ~BoidsSim();

    // Change number of boids
    void resize(std::uint32_t num_boids);

    // Setters and getters for sim parameters
    void set_params(const BoidsParams& p);
    BoidsParams get_params() const;

    // Get number of boids
    std::uint32_t count() const;

    // Step simulation forward one frame in time
    void step(float dt);

    // Get the boid position/velocities
    std::vector<float> positions_host() const;   // returns [x0..xN-1, y0..yN-1]
    std::vector<float> velocities_host() const;  // returns [vx0..vxN-1, vy0..vyN-1]

private:
    void allocate(std::uint32_t n);
    void free(); // Deallocate all boid state parameters
    void ensure_grid_buffers();  // new: allocate grid-related buffers

    std::uint32_t N_; // Number of boids
    BoidsParams params_; // Simulation parameters

    // Device arrays for state
    float* d_posx_;
    float* d_posy_;
    float* d_velx_;
    float* d_vely_;

    // RNG state (new, for curand)
    void* d_rng_; // stored as opaque pointer to curand state type in .cu
};
