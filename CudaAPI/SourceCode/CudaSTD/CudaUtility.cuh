#ifndef __CUDAUYILITY__CUH__
#define __CUDAUYILITY__CUH__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/std/cassert>
#include <device_launch_parameters.h>

#define Int int
#define Uint unsigned int
#define Float double

//CHECK(bool isAssert, char* errorInfo);
//This marco can run on host and device.
//If isAssert is false, then the program will stop.
//If isAssert is true, nothing will be happen.
#define CHECK(isAssert, errorInfo) check(isAssert, errorInfo, __FILE__, __LINE__)
//Do not use this function, recommend marco CHECK to use.
__host__ __device__ void check(const bool& isAssert, const char* errorInfo, const char* file, const int& line);


//CUDA_CHECK(cudaError_t err);
//This marco can run only on host.
//This macro is used to capture error and print error information when call CUDA functions.
#define ERROR_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)
//Do not use this function, recommend marco ERROR_CHECK to use.
void error_check(cudaError_t err, const char* file, int line);

#endif // !__CUDAUYILITY__CUH__
