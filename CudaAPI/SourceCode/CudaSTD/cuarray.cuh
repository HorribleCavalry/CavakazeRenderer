#ifndef __CUARRAY__CUH__
#define __CUARRAY__CUH__

namespace custd
{
	template<typename T, int size>
	class array
	{
	private:
		T data[size];
	public:
		__host__ __device__ T();
		__host__ __device__ T(const array<T,size>& ar);
	};
}

#endif // !__CUARRAY__CUH__S
