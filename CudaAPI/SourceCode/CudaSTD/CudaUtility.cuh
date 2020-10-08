#pragma once
//#ifndef __CUDAUYILITY__CUH__
//#define __CUDAUYILITY__CUH__

#include <math.h>
#include <cstdio>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda/std/cassert>
#include <device_launch_parameters.h>

typedef int Int;
typedef bool Bool;
typedef double Float;
typedef unsigned int Uint;
typedef unsigned short Ushort;

#define __duel__ __host__ __device__

#define RUN_ON_DEVICE

//CHECK(bool isAssert, char* errorInfo);
//This marco can run on host and device.
//If isAssert is false, then the program will stop.
//If isAssert is true, nothing will be happen.
#define CHECK(isAssert, errorInfo) check(isAssert, errorInfo, __FILE__, __LINE__)
//Try to avoid using this function, recommend marco CHECK to use.
__duel__ void check(const bool& isAssert, const char* errorInfo, const char* file, const int& line);

//CHECK(bool isAssert, char* errorInfo, Number val);
//This marco can run on host and device.
//If isAssert is false, then the program will stop.
//If isAssert is true, nothing will be happen.
//#define CHECK(isAssert, errorInfo, val) check(isAssert, errorInfo, val, __FILE__, __LINE__)
//Try to avoid using this function, recommend marco CHECK to use.
__duel__ void check(const bool& isAssert, const char* errorInfo, const Bool& val, const char* file, const int& line);
__duel__ void check(const bool& isAssert, const char* errorInfo, const Int& val, const char* file, const int& line);
__duel__ void check(const bool& isAssert, const char* errorInfo, const Uint& val, const char* file, const int& line);
__duel__ void check(const bool& isAssert, const char* errorInfo, const Float& val, const char* file, const int& line);


//#define CHECK(isAssert, errorInfo, valArray, size) check(isAssert, errorInfo, valArray, size, __FILE__, __LINE__)

//__duel__ void check(const bool& isAssert, const char* errorInfo, const Int val[], const Uint& size, const char* file, const int& line);


//CUDA_CHECK(cudaError_t err);
//This marco can run only on host.
//This macro is used to capture error and print error information when call CUDA functions.
#define ERROR_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)
//Do not use this function, recommend marco ERROR_CHECK to use.
void error_check(cudaError_t err, const char* file, int line);

template<typename T, typename U, typename K>
const Bool IsInRange(const T& n, const U& minN, const K& maxN)
{
	return n >= minN && n <= maxN;
}

template<typename T>
__global__ void ApplyDeviceVirtualPtr(T* devicePtr)
{
	T temp(*devicePtr);
	T* tempPtr = &temp;
	Int insBytes = sizeof(T);
	void* desUnit = (void*)devicePtr;
	memcpy(devicePtr, tempPtr, sizeof(T));
}

template<typename T>
__host__ T* CudaInsMemCpyHostToDevice(const T* insWithDevicePtr)
{
	T* device;
	cudaMalloc(&device, sizeof(T));
	cudaMemcpy(device, insWithDevicePtr, sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
	ApplyDeviceVirtualPtr << <1, 1 >> > (device);
	return device;
}

#ifdef RUN_ON_DEVICE
__device__ static curandState* deviceStates;

__host__ void InitDeviceStates(const Int& length);

__device__ Float GetUniformRand();
#endif // RUN_ON_DEVICE

#ifdef RUN_ON_HOST
__host__ Float GetUniformRand();
#endif // RUN_ON_HOST

//#endif // !__CUDAUYILITY__CUH__
