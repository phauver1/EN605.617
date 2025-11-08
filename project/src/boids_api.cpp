/*
    C++ side of the Python/C++ API 
*/

#include "boids.hpp"
#include <cstring>
#include <iostream>

extern "C" {
    // thread_local string to save error messages
    static thread_local std::string error_string;

    // Return current error string
    const char* boids_error_string() {
        return error_string.c_str();
    }

    // Allocates a new BoidsSim object
    BoidsSim* boids_create(std::uint32_t num_boids, BoidsParams p, std::uint64_t seed) {
        try {return new BoidsSim(num_boids, p, seed);}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
            return nullptr;
        }
    }

    // Destroys a BoidSim object
    void boids_destroy(BoidsSim* sim) {
        try {delete sim;}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
        }
    }

    // Step the BoidsSim object
    void boids_step(BoidsSim* sim, float dt) {
        try {sim->step(dt);}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
        }
    }

    // Get the number of boids in the sim
    std::uint32_t boids_count(BoidsSim* sim) {
        try {return sim->count();}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
            return 0;
        }
    }

    // Get the positions of all boids in the sim
    void boids_positions(BoidsSim* sim, float* out) {
        try {
            auto pos = sim->positions_host();
            std::memcpy(out, pos.data(), pos.size() * sizeof(float));
        }
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
        }
    }

    // Get the velocities of all boids in the sim
    void boids_velocities(BoidsSim* sim, float* out) {
        try {
            auto vel = sim->velocities_host();
            std::memcpy(out, vel.data(), vel.size() * sizeof(float));
        }
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
        }
    }

    // Retrieve the BoidsSim parameters
    BoidsParams boids_get_params(BoidsSim* sim) {
        try {return sim->get_params();}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
            return BoidsParams{};
        }
    }

    // Set the BoidsSim parameters
    void boids_set_params(BoidsSim* sim, BoidsParams p) {
        try {sim->set_params(p);}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
        }
    }
    
    // Change the number of boids in the sim
    void boids_resize(BoidsSim* sim, std::uint32_t new_count) {
        try {sim->resize(new_count);}
        catch (const std::exception& e) {
            error_string += std::string("\n") + e.what();
        }
    }
}
