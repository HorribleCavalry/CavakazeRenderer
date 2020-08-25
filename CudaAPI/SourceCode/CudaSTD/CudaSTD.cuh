#ifndef __CUDASTD__CUH__
#define __CUDASTD__CUH__
#include "../Common/CudaUtility.cuh"
namespace custd
{
	template<typename T>
	__host__ __device__ void print(const T& para);

	class Stream
	{
	public:
		__host__ __device__ Stream();
		__host__ __device__ Stream(const Stream& st) = delete;
		__host__ __device__ Stream(Stream&& st) = delete;
		__host__ __device__ Stream& operator=(const Stream& st) = delete;
		__host__ __device__ Stream& operator=(Stream&& st) = delete;
		__host__ __device__ ~Stream();
		const Stream& operator<<(const short& val) const;
		const Stream& operator<<(const unsigned short& val) const;

		const Stream& operator<<(const int& val) const;
		const Stream& operator<<(const unsigned int& val) const;

		const Stream& operator<<(const long& val) const;
		const Stream& operator<<(const unsigned long& val) const;

		const Stream& operator<<(const long long& val) const;
		const Stream& operator<<(const unsigned long long& val) const;

		const Stream& operator<<(const float& val) const;
		const Stream& operator<<(const double& val) const;

		const Stream& operator<<(const char& val) const;
		const Stream& operator<<(const char* val) const;

		const Stream& operator<<(void (*edl)()) const;

	};
	const Stream cout;
	__host__ __device__ void endl();
}

#endif // !__CUDASTD__CUH__