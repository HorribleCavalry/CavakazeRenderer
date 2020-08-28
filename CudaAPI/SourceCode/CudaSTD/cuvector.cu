#include "cuvector.cuh"

namespace custd
{

	template<typename T>
	__host__ __device__ Uint cuvector<T>::size()
	{
		return Size;
	}
	template<typename T>
	__host__ __device__ Uint cuvector<T>::capacity()
	{
		return Capacity;
	}

}