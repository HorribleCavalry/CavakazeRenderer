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

void error_check(cudaError_t err, const char * file, int line)
{
	if (err != cudaSuccess) {
		::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}
}