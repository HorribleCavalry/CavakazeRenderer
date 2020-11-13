#ifndef __CUDAAPI__CUH__
#define __CUDAAPI__CUH__

#define CUDA_API __declspec(dllexport)

extern "C"
{
	CUDA_API void StartRendering();
}
#endif // !__CUDAAPI__CUH__