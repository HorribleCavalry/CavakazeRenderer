#ifndef __CUDASTD__CUH__
#define __CUDASTD__CUH__
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/std/cassert>
#include <device_launch_parameters.h>

namespace custd
{
	template<typename T>
	__host__ __device__ void print(const T& para);

	class Stream
	{
	public:
		__host__ __device__ Stream();
		//__host__ __device__ Stream(const Stream&) = delete;
		//__host__ __device__ Stream(Stream&&) = delete;
		//__host__ __device__ Stream& operator=(const Stream&) = delete;
		//__host__ __device__ Stream& operator=(Stream&&) = delete;
		__host__ __device__ ~Stream();
		__host__ __device__ const Stream& operator<<(const short& val) const;
		__host__ __device__ const Stream& operator<<(const unsigned short& val) const;

		__host__ __device__ const Stream& operator<<(const int& val) const;
		__host__ __device__ const Stream& operator<<(const unsigned int& val) const;

		__host__ __device__ const Stream& operator<<(const long& val) const;
		__host__ __device__ const Stream& operator<<(const unsigned long& val) const;

		__host__ __device__ const Stream& operator<<(const long long& val) const;
		__host__ __device__ const Stream& operator<<(const unsigned long long& val) const;

		__host__ __device__ const Stream& operator<<(const float& val) const;
		__host__ __device__ const Stream& operator<<(const double& val) const;

		__host__ __device__ const Stream& operator<<(const char& val) const;
		__host__ __device__ const Stream& operator<<(const char* val) const;

		__host__ __device__ const Stream& operator<<(void(*edl)()) const;

	};

	const Stream cout;
	 //const Stream kout;
	__host__ __device__ void endl();
	
	namespace
	{
		class InitKernelResource
		{
		public:
			InitKernelResource();
			InitKernelResource(const InitKernelResource&) = delete;
			InitKernelResource(InitKernelResource&&) = delete;
			InitKernelResource& operator=(const InitKernelResource&) = delete;
			InitKernelResource& operator=(InitKernelResource&&) = delete;
			~InitKernelResource();
		};
		static const InitKernelResource initCout;
		InitKernelResource::InitKernelResource()
		{
			//cudaMemcpyToSymbol(custd::cout, &cout, sizeof(Stream));
		}

		InitKernelResource::~InitKernelResource()
		{
		}
	}
}

#endif // !__CUDASTD__CUH__