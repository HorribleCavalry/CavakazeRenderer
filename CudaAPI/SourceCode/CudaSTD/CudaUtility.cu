#include "CudaUtility.cuh"
void check(const bool & isAssert, const char * errorInfo, const char * file, const int & line)
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

__host__ __device__ void check(const bool & isAssert, const char * errorInfo, const Bool & val, const char * file, const int & line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info: %s\n", errorInfo);
		const char* trueStr = "true";
		const char* falseStr = "false";
		const char* boolVal = val ? trueStr : falseStr;
		printf("Error Int val: %s\n", boolVal);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

__host__ __device__ void check(const bool & isAssert, const char * errorInfo, const Int & val, const char * file, const int & line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info: %s\n", errorInfo);
		printf("Error Int val: %d\n", val);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

__host__ __device__ void check(const bool & isAssert, const char * errorInfo, const Uint & val, const char * file, const int & line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info: %s\n", errorInfo);
		printf("Error Int val: %u\n", val);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

__host__ __device__ void check(const bool & isAssert, const char * errorInfo, const Float & val, const char * file, const int & line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info: %s\n", errorInfo);
		printf("Error Float val: %f\n", val);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

__host__ __device__ void check(const bool & isAssert, const char * errorInfo, const Int* valArry, const Uint & size, const char * file, const int & line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info: %s\n", errorInfo);
		for (Uint i = 0; i < size; i++)
		{
			printf("Error Int valArray[%u]: %d\n", i,valArry[i]);
		}
		if (valArry) delete[]valArry;
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

void error_check(cudaError_t err, const char * file, int line)
{
	if (err != cudaSuccess) {
		::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}
}