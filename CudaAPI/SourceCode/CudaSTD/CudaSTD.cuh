#pragma once
#include "../Common/CudaUtility.cuh"
namespace custd
{
	template<typename T>
	static __host__ __device__ void print(const T& para);
}