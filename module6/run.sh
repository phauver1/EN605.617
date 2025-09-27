echo "Building Code"
nvcc assignment.cu -o assignment

echo "Running executible"
./assignment "$@"

echo "Running executible"
./assignment 2560 256

echo "Running executible"
./assignment 2560 128

echo "Running executible"
./assignment 1920 192
