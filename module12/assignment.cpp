

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline bool
readCommandLineInputs(int argc, char** argv, int* platform, bool* useMap) {
    for (int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);

        if (!input.compare("--platform"))
        {
            input = std::string(argv[++i]);
            std::istringstream buffer(input);
            buffer >> *platform;
        }
        else if (!input.compare("--useMap"))
        {
            *useMap = true;
        }
        else
        {
            std::cout << "usage: --platform n --useMap" << std::endl;
            return false;
        }
    }
    return true;
}

inline void
createInputVectors(cl_uint numDevices, 
                    float** aVector, 
                    cl_float4** xVector, 
                    cl_float4** yVector) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    *xVector = new cl_float4[NUM_BUFFER_ELEMENTS * numDevices];
    *yVector = new cl_float4[NUM_BUFFER_ELEMENTS * numDevices];
    *aVector = new float[NUM_BUFFER_ELEMENTS * numDevices];

    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        (*aVector)[i] = dist(gen);
        (*xVector)[i] = {dist(gen), dist(gen), dist(gen), dist(gen)};
        (*yVector)[i] = {dist(gen), dist(gen), dist(gen), dist(gen)};
    }
}

inline void
createBuffersAndSubbuffers(int numDevices,
                            cl_mem& main_buffer,
                            std::vector<cl_mem>& buffers,
                            cl_context context) {
    // create buffers and sub-buffers
    cl_int errNum;

    // create a single buffer to cover all the input data
    main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        (sizeof(float)+2*sizeof(cl_float4)) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // now for all devices other than the first create a sub-buffer
    for (unsigned int i = 0; i < numDevices; i++)
    {
        cl_buffer_region region = 
            {
                NUM_BUFFER_ELEMENTS * i * (sizeof(float)+2*sizeof(cl_float4)), 
                NUM_BUFFER_ELEMENTS * (sizeof(float)+2*sizeof(cl_float4))
            };
        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }
}

inline void
enqueMaps(int numDevices,
            std::vector<cl_command_queue>& queues,
            float* aVector,
            cl_float4* xVector,
            cl_float4* yVector,
            cl_mem& main_buffer){
    size_t totalSize = (sizeof(float) + 2 * sizeof(cl_float4)) * NUM_BUFFER_ELEMENTS * numDevices;
    cl_int errNum;

    void* mapPtr = clEnqueueMapBuffer(
        queues[numDevices - 1], main_buffer, CL_TRUE, CL_MAP_WRITE,
        0, totalSize, 0, NULL, NULL, &errNum);
    checkErr(errNum, "clEnqueueMapBuffer(..)");
    cl_float* aMapPtr = reinterpret_cast<cl_float*>(mapPtr);
    cl_float4* xMapPtr = reinterpret_cast<cl_float4*>(aMapPtr + NUM_BUFFER_ELEMENTS * numDevices);
    cl_float4* yMapPtr = reinterpret_cast<cl_float4*>(xMapPtr + NUM_BUFFER_ELEMENTS * numDevices);
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++) {
        aMapPtr[i] = aVector[i];
        xMapPtr[i] = xVector[i];
        yMapPtr[i] = yVector[i];
    }
    errNum = clEnqueueUnmapMemObject(
        queues[numDevices - 1], main_buffer, mapPtr, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueUnmapMemObject(..)");
}

inline void
enqueWriteBuffers(int numDevices,
            std::vector<cl_command_queue>& queues,
            float* aVector,
            cl_float4* xVector,
            cl_float4* yVector,
            cl_mem& main_buffer){
    cl_int errNum;

    size_t aSize = sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices;
    size_t xSize = sizeof(cl_float4) * NUM_BUFFER_ELEMENTS * numDevices;
    size_t ySize = sizeof(cl_float4) * NUM_BUFFER_ELEMENTS * numDevices;

    // Write aVector at offset 0
    errNum = clEnqueueWriteBuffer( queues[numDevices - 1], main_buffer,
        CL_TRUE, 0, aSize, (void*)aVector, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueWriteBuffer(aVector)");

    // Write xVector immediately after aVector
    errNum = clEnqueueWriteBuffer( queues[numDevices - 1], main_buffer,
        CL_TRUE, aSize, xSize, (void*)xVector, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueWriteBuffer(xVector)");

    // Write yVector immediately after xVector
    errNum = clEnqueueWriteBuffer( queues[numDevices - 1], main_buffer,
        CL_TRUE, aSize + xSize, ySize, (void*)yVector, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueWriteBuffer(yVector)");
}

inline void
writeOutput(int numDevices, cl_float4* yVector) {
    // Display output in rows
    for (unsigned i = 0; i < numDevices; i++)
    {
        for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i+1) * NUM_BUFFER_ELEMENTS); elems++)
        {
            std::cout << "(" 
                << yVector[elems].s[0] << ", "
                << yVector[elems].s[1] << ", "
                << yVector[elems].s[2] << ", "
                << yVector[elems].s[3] << ")" 
                << std::endl;
        }

        std::cout << std::endl;
    }
}

inline void
readOutput(int numDevices, bool useMap, cl_float4* yVector,
    std::vector<cl_command_queue>& queues, cl_mem& main_buffer) {
    cl_int errNum;
    if (useMap)
    {
        cl_float4 * yMapPtr = (cl_float4*) clEnqueueMapBuffer(
            queues[numDevices - 1], main_buffer, CL_TRUE, CL_MAP_READ, 0,
            sizeof(cl_float4) * NUM_BUFFER_ELEMENTS * numDevices, 0,
            NULL, NULL, &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(..)");

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++) {
            yVector[i] = yMapPtr[i];
        }

        errNum = clEnqueueUnmapMemObject( queues[numDevices - 1], main_buffer,
            yMapPtr, 0, NULL, NULL);
        clFinish(queues[numDevices - 1]);
    }
    else 
    {
        // Read back computed data
        clEnqueueReadBuffer( queues[numDevices - 1], main_buffer, CL_TRUE,
            0, sizeof(cl_float4) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)yVector, 0, NULL, NULL);
    }
}

inline void
createCommandQueues(int numDevices, cl_context context, std::vector<cl_command_queue>& queues,
    cl_device_id* deviceIDs, std::vector<cl_mem>& buffers, cl_program program,
    std::vector<cl_kernel>& kernels, uint numBufferElements) {
    cl_int errNum;
    // Create command queues
    for (unsigned int i = 0; i < numDevices; i++)
    {
        InfoDevice<cl_device_type>::display(
            deviceIDs[i], 
            CL_DEVICE_TYPE, 
            "CL_DEVICE_TYPE");

        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[i],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "bufferSaxpy",
            &errNum);
        checkErr(errNum, "clCreateKernel(bufferSaxpy)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        errNum = clSetKernelArg(kernel, 1, sizeof(uint), &numBufferElements);
        checkErr(errNum, "clSetKernelArg(bufferSaxpy)");

        kernels.push_back(kernel);
    }
}

inline std::string
getDevices(int platform, cl_uint numPlatforms, cl_platform_id*& platformIDs,
        cl_device_id*& deviceIDs, cl_uint& numDevices) {
    cl_int errNum;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs");
    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    platformIDs = new cl_platform_id[numPlatforms];
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

    std::string srcProg( std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    DisplayPlatformInfo(platformIDs[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, 
        0, NULL, reinterpret_cast<cl_uint*>(&numDevices));
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND) {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = new cl_device_id[numDevices];
    errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL,
        numDevices, deviceIDs, NULL);
    checkErr(errNum, "clGetDeviceIDs");

    return srcProg;
}

inline void
buildContexAndProgram(int platform, cl_uint numPlatforms, cl_platform_id*& platformIDs,
    cl_device_id*& deviceIDs, cl_uint& numDevices, cl_context& context, cl_program& program) {
    cl_int errNum;

    std::string srcProg = getDevices(platform, numPlatforms, platformIDs, deviceIDs, numDevices);

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext( contextProperties, numDevices,
        deviceIDs, NULL, NULL, &errNum);
    checkErr(errNum, "clCreateContext");
    
    // Create program from source
    program = clCreateProgramWithSource(context, 1, 
        &src, &length, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(program, numDevices, deviceIDs,
        "-I.", NULL, NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo( program, deviceIDs[0], CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

}

inline void
buildAndRun(int platform, bool useMap) {

    cl_mem main_buffer;
    cl_float4 * xVector;
    cl_float4 * yVector;
    float * aVector;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    cl_int errNum;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_uint numPlatforms, numDevices;

    buildContexAndProgram(platform, numPlatforms, platformIDs,
        deviceIDs, numDevices, context, program);
    createInputVectors(numDevices, &aVector, &xVector, &yVector);
    createBuffersAndSubbuffers(numDevices, main_buffer, buffers, context);
    createCommandQueues(numDevices,context,queues,deviceIDs,buffers,program,kernels
        ,NUM_BUFFER_ELEMENTS);
    if (useMap) {enqueMaps(numDevices, queues, aVector, xVector, yVector, main_buffer);}
    else {enqueWriteBuffers(numDevices, queues, aVector, xVector, yVector, main_buffer);}

    std::vector<cl_event> events;
    // call kernel for each device
    for (unsigned int i = 0; i < queues.size(); i++) {
        cl_event event;
        size_t gWI = NUM_BUFFER_ELEMENTS;
        errNum = clEnqueueNDRangeKernel(queues[i], kernels[i], 1, NULL,
            (const size_t*)&gWI, (const size_t*)NULL, 0, 0, &event);
        events.push_back(event);
    }

    clWaitForEvents(events.size(), &events[0]);
    readOutput(numDevices, useMap, yVector, queues, main_buffer);
    writeOutput(numDevices, yVector);
}

int main(int argc, char** argv)
{
    int platform = DEFAULT_PLATFORM; 
    bool useMap  = DEFAULT_USE_MAP;

    if (!readCommandLineInputs(argc, argv, &platform, &useMap)) return 0;

    buildAndRun(platform, useMap);

    std::cout << "Program completed successfully" << std::endl;
    return 0;
}