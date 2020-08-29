﻿#ifndef __CUDAUYILITY__CUH__
#define __CUDAUYILITY__CUH__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/std/cassert>
#include <device_launch_parameters.h>

#define Bool bool
#define Int int
#define Uint unsigned int
#define Float double

#define __duel__ __host__ __device__

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

__duel__ void check(const bool& isAssert, const char* errorInfo, const Int val[], const Uint& size, const char* file, const int& line);


//CUDA_CHECK(cudaError_t err);
//This marco can run only on host.
//This macro is used to capture error and print error information when call CUDA functions.
#define ERROR_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)
//Do not use this function, recommend marco ERROR_CHECK to use.
void error_check(cudaError_t err, const char* file, int line);

#endif // !__CUDAUYILITY__CUH__
