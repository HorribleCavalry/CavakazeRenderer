#ifndef __CUDAAPI__CUH__
#define __CUDAAPI__CUH__
#include <Windows.h>

#define CUDA_API __declspec(dllexport)

extern "C"
{
	CUDA_API void OpenDebugConsole();
	CUDA_API void CloseDebugConsole();

	CUDA_API void InitializeResources();
	CUDA_API void PrepareScene();
	CUDA_API void StartRendering();
	CUDA_API void ImageCopyToForm();
}
#endif // !__CUDAAPI__CUH__