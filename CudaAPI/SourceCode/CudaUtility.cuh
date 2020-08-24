#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/std/cassert>
#include <device_launch_parameters.h>

#define CHECK(isAssert, errorInfo) check(isAssert, errorInfo, __FILE__, __LINE__)
__device__ void check(const bool& isAssert, const char* errorInfo, const char* file, const int& line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info:%s\n", errorInfo);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

#define CUDA_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)
inline void error_check(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}
}