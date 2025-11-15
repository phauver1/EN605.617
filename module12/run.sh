rm ./assignment
g++ assignment.cpp -o assignment -w -lOpenCL
./assignment "$@" 
