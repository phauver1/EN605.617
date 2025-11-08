rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_FLAGS="-G -lineinfo"
make -j
cp libcuda_boids.so ../
