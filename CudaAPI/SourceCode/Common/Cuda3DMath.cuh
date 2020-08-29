#ifndef __CUDA3DMATH__CUH__
#define __CUDA3DMATH__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "../CudaSTD/cuvector.cuh"

template<typename T>
class vec2
{
public:
	T x, y;
public:
	vec2();
};

//typedef vec2i vec2<Int>;

#endif // !__CUDA3DMATH__CUH__
