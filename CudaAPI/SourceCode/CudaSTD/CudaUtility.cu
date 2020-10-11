#include "CudaUtility.cuh"
__duel__ void check(const Bool & isNotAssert, const char * errorInfo, const char * file, const int & line)
{
	if (!isNotAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info:%s\n", errorInfo);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

__duel__ void check(const Bool & isAssert, const char * errorInfo, const Bool & val, const char * file, const int & line)
{
	if (!isAssert)
	{
		printf("====================Error Occurred!====================\n");
		printf("Error occurred in: %s\n", file);
		printf("Line: %d\n", line);
		printf("Error info: %s\n", errorInfo);
		const char* trueStr = "true";
		const char* falseStr = "false";
		const char* BoolVal = val ? trueStr : falseStr;
		printf("Error Int val: %s\n", BoolVal);
		printf("====================Error Occurred!====================\n");
		assert(false);
	}
}

__duel__ void check(const Bool & isAssert, const char * errorInfo, const Int & val, const char * file, const int & line)
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

__duel__ void check(const Bool & isAssert, const char * errorInfo, const Uint & val, const char * file, const int & line)
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

__duel__ void check(const Bool & isAssert, const char * errorInfo, const Float & val, const char * file, const int & line)
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

__duel__ void check(const Bool & isAssert, const char * errorInfo, const Int* valArry, const Uint & size, const char * file, const int & line)
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

#ifdef RUN_ON_DEVICE

__global__ void SetupDeviceStates()
{
	Int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(10086, globalIdx, 0, &deviceStates[globalIdx]);
}

__host__ void InitDeviceStates(const Int& length)
{
	curandState* deviceStatesH = nullptr;
	if (deviceStatesH)
	{
		cudaFree(deviceStatesH);
		deviceStatesH = nullptr;
	}
	cudaMalloc(&deviceStatesH, length * sizeof(curandState));

	cudaMemcpyToSymbol(deviceStates, &deviceStatesH, sizeof(curandState*));

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	const Int threadNum = 32;
	Int blockNum = length / threadNum;
	SetupDeviceStates << <blockNum, threadNum >> > ();

	error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
}

__device__ Float GetUniformRand()
{
	const Int& globalIdx = blockIdx.x*blockDim.x + threadIdx.x;
	curandState& localState = deviceStates[globalIdx];
	return curand_uniform(&localState);
}
#endif // RUN_ON_DEVICE

#ifdef RUN_ON_HOST
__host__ Float GetUniformRand()
{
	static std::default_random_engine randEngine(rand());
	static std::uniform_real_distribution<Float> randGenerator(0.0, 1.0);
	return randGenerator(randEngine);
}
#endif // RUN_ON_HOST