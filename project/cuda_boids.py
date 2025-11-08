#
# Python side of the Python/C++ API
#

import ctypes
import os
import sys

# Load shared library we compiled
if sys.platform.startswith("win"): libname = "cuda_boids.dll"
else:                              libname = "libcuda_boids.so"

# Get the path of the shared library
libpath = os.path.join(os.path.dirname(__file__), libname)
# Load the library
cuda = ctypes.CDLL(libpath)

# Define BoidsParams struct (matches C++ layout)
class BoidsParams(ctypes.Structure):
    _fields_ = [
        ("perception_radius", ctypes.c_float),
        ("angle_limit", ctypes.c_float),
        ("max_speed", ctypes.c_float),
        ("hysteresis", ctypes.c_float),
        ("align_weight", ctypes.c_float),
        ("cohese_weight", ctypes.c_float),
        ("separate_weight", ctypes.c_float),
        ("boundary_weight", ctypes.c_float),
        ("world_width", ctypes.c_float),
        ("world_height", ctypes.c_float)
    ]

# Define opaque BoidsSim pointer
class BoidsSim(ctypes.Structure):
    pass

BoidsSim_p = ctypes.POINTER(BoidsSim)

#
# Bind functions
#

# Create BoidsSim object
cuda.boids_create.argtypes = [ctypes.c_uint32, BoidsParams, ctypes.c_uint64]
cuda.boids_create.restype = BoidsSim_p

# Destroy BoidsSim object
cuda.boids_destroy.argtypes = [BoidsSim_p]
cuda.boids_destroy.restype = None

# Step the BoidsSim forward one frame
cuda.boids_step.argtypes = [BoidsSim_p, ctypes.c_float]
cuda.boids_step.restype = None

# Get number of boids objects in the sim
cuda.boids_count.argtypes = [BoidsSim_p]
cuda.boids_count.restype = ctypes.c_uint32

# Get the positions of the boids for drawing purposes
cuda.boids_positions.argtypes = [BoidsSim_p, ctypes.POINTER(ctypes.c_float)]
cuda.boids_positions.restype = None

# Get the velocity of boids for drawing purposes
cuda.boids_velocities.argtypes = [BoidsSim_p, ctypes.POINTER(ctypes.c_float)]
cuda.boids_velocities.restype = None

# Get BoidsSim parameters
cuda.boids_get_params.argtypes = [BoidsSim_p]
cuda.boids_get_params.restype = BoidsParams

# Get Boids API error string
cuda.boids_error_string.argtypes = []          # no arguments
cuda.boids_error_string.restype = ctypes.c_char_p  # returns const char*

# Wrapper for the BoidsSim object
class PyBoidsSim:
    # Construct the object
    def __init__(self, num_boids, params, seed=1234):
        self.obj = cuda.boids_create(num_boids, params, seed)
        if not self.obj:
            raise RuntimeError("Failed to create BoidsSim")

    # Delete the object
    def __del__(self):
        if hasattr(self, "obj") and self.obj:
            cuda.boids_destroy(self.obj)

    # Explicit destructor
    def close(self):
        if self.obj:
            cuda.boids_destroy(self.obj)
            self.obj = None

    # Step the Simulation forward one frame
    def step(self, dt):
        cuda.boids_step(self.obj, dt)

    # Get the number of boids in the sim
    def count(self):
        return cuda.boids_count(self.obj)

    # Get the positions of all the boids for drawing purposes
    def positions_host(self):
        N = self.count()
        arr = (ctypes.c_float * (2 * N))()
        cuda.boids_positions(self.obj, arr)
        return list(arr)

    # Get the velocities of all the boids for drawing purposes
    def velocities_host(self):
        N = self.count()
        arr = (ctypes.c_float * (2 * N))()
        cuda.boids_velocities(self.obj, arr)
        return list(arr)

    # Get the simulation parameters
    def get_params(self):
        return cuda.boids_get_params(self.obj)
    
    # Set the simulation parameters
    def set_params(self, params):
        cuda.boids_set_params(self.obj, params)

    # Set the number of boids in the simulation
    def resize(self, new_count):
        cuda.boids_resize(self.obj, new_count)

