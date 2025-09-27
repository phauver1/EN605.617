echo "Building Code"
nvcc assignment.cu -o assignment
echo "Running executible"
./assignment "$@"
