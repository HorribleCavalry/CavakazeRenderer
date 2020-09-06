#ifndef __CUDASTD__CUH__
#define __CUDASTD__CUH__
#include "CudaUtility.cuh"

namespace custd
{
	template<typename T>
	__duel__ void print(const T& para);

	class Ostream
	{
	public:
		__duel__ Ostream();
		__duel__ ~Ostream();
		__duel__ const Ostream& operator<<(const short& val) const;
		__duel__ const Ostream& operator<<(const unsigned short& val) const;

		__duel__ const Ostream& operator<<(const int& val) const;
		__duel__ const Ostream& operator<<(const unsigned int& val) const;

		__duel__ const Ostream& operator<<(const long& val) const;
		__duel__ const Ostream& operator<<(const unsigned long& val) const;

		__duel__ const Ostream& operator<<(const long long& val) const;
		__duel__ const Ostream& operator<<(const unsigned long long& val) const;

		__duel__ const Ostream& operator<<(const float& val) const;
		__duel__ const Ostream& operator<<(const double& val) const;

		__duel__ const Ostream& operator<<(const char& val) const;
		__duel__ const Ostream& operator<<(const char* val) const;

		__duel__ const Ostream& operator<<(void(*edl)()) const;
	};

	const Ostream cout;

	//template<typename T>
	__device__ void kout(const int& val);
	 //const Stream kout;
	__duel__ void endl();
	//__device__ float deviceTest;
	namespace
	{
		const Ostream cout;
		class InitKernelResource
		{
		public:
			InitKernelResource();
			InitKernelResource(const InitKernelResource&) = delete;
			InitKernelResource(InitKernelResource&&) = delete;
			InitKernelResource& operator=(const InitKernelResource&) = delete;
			InitKernelResource& operator=(InitKernelResource&&) = delete;
			~InitKernelResource();
		};
		static const InitKernelResource initCout;
		InitKernelResource::InitKernelResource()
		{
			//float val = 4.0f;
			//cudaMemcpyToSymbol(deviceTest, &val, sizeof(float));
		}

		InitKernelResource::~InitKernelResource()
		{
		}
	}
}

#endif // !__CUDASTD__CUH__