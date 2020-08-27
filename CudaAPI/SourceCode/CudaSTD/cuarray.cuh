#ifndef __CUARRAY__CUH__
#define __CUARRAY__CUH__

#include "CudaUtility.cuh"

namespace custd
{
	template<typename T, int size>
	class cuarray
	{
	private:
		T* data;
	public:
		__host__ __device__ cuarray();
		__host__ __device__ cuarray(const cuarray<T, size>& ar) = delete;
		__host__ __device__ cuarray(cuarray<T, size>&& ar) = delete;
	};

}

#endif // !__CUARRAY__CUH__S
