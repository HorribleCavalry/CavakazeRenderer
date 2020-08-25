#ifndef __CUDASTD__CUH__
#define __CUDASTD__CUH__
#include "../Common/CudaUtility.cuh"
namespace custd
{
	template<typename T>
	__host__ __device__ void print(const T& para);
}

#endif // !__CUDASTD__CUH__