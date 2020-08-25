#include "CudaAPI.cuh"
#include <limits>
#include "CudaSTD/CudaSTD.cuh"
#include "Common/CudaUtility.cuh"

#include <cuda/std/type_traits>
#include <cuda/std/utility>

class ost
{
	__host__ __device__ ost()
	{
		printf("Called ost\n");
	}

	__host__ __device__ ost(const ost&)
	{
		printf("Called ost\n");
	}

	__host__ __device__ ost(ost&&)
	{
		printf("Called ost\n");
	}
};

__global__ void kernel()
{
	ost ot;
	//custd::Stream cout;
	//cout << "Yes";
	//cuda::std::
	//custd::print(3);
	//custd::Stream st;
	//st << "Yes\n";
	printf("Yes");
}

int main()
{
	short shortNum = std::numeric_limits<short>::max();
	unsigned short unsignedShortNum = std::numeric_limits<unsigned short>::max();

	int intNum = std::numeric_limits<int>::max();
	unsigned int unsignedIntNum = std::numeric_limits<unsigned int>::max();

	long longNum = std::numeric_limits<long>::max();
	unsigned long unsignedLongNum = std::numeric_limits<unsigned long>::max();

	long long longLongNum = std::numeric_limits<long long>::max();
	unsigned long long unsignedLongLongNum = std::numeric_limits<unsigned long long>::max();

	float floatNum = std::numeric_limits<float>::max();
	double doubleNum = std::numeric_limits<double>::max();

	custd::cout << "shortNum: " << shortNum <<custd::endl;
	custd::cout << "unsignedShortNum: " << unsignedShortNum <<custd::endl;

	custd::cout << "intNum: " << intNum <<custd::endl;
	custd::cout << "unsignedIntNum: " << unsignedIntNum <<custd::endl;

	custd::cout << "longNum: " << longNum <<custd::endl;
	custd::cout << "unsignedLongNum: " << unsignedLongNum <<custd::endl;

	custd::cout << "longLongNum: " << longLongNum <<custd::endl;
	custd::cout << "unsignedLongLongNum: " << unsignedLongLongNum <<custd::endl;

	custd::cout << "floatNum: " << floatNum << custd::endl;
	custd::cout << "doubleNum: " << doubleNum << custd::endl;

	custd::cout << "Now the following code is called in kernel:" << custd::endl;
	//custd::initCout
	kernel << <1, 1 >> > ();
}