#ifndef __CUVACTOR__CUH__
#define __CUVACTOR__CUH__

#include "CudaUtility.cuh"


namespace custd
{
	template<typename T>
	class cuvector
	{
	private:
		T* data;
		Uint Size;
		Uint BackIdx;
		Uint Capacity;
	public:
		__host__ __device__ cuvector();
		__host__ __device__ ~cuvector();

		__host__ __device__ Uint size();
		__host__ __device__ Uint capacity();

		void push_back(const T& val);

		T& operator[](Int idx);

	};
}

#endif // !__CUVACTOR__CUH__
