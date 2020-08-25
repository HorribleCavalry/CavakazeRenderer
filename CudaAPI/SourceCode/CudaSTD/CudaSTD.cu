#include "CudaSTD.cuh"
#include <cuda/std/type_traits>
namespace custd
{
	template<typename T>
	__host__ __device__ void print(const T & para)
	{
		//switch (typeid(para))
		//{
		//	case typeid(short) : printf("%d", para); printf("\nCalled printf(short)\n"); break;
		//	case typeid(unsigned short) : printf("%d", para); printf("\nCalled printf(unsigned short)\n"); break;

		//	case typeid(int) : printf("%d", para); printf("\nCalled printf(int)\n"); break;
		//	case typeid(unsigned int) : printf("%d", para); printf("\nCalled printf(unsigned int)\n"); break;

		//	case typeid(float) : printf("%d", para); printf("\nCalled printf(float)\n"); break;

		//	case typeid(double) : printf("%d", para); printf("\nCalled printf(double)\n"); break;
		//default: printf("\nUnknown type.\n"); break;
		//}
		printf("Nothing");
	}
}