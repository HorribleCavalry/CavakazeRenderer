#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda/std/cassert>
namespace std
{
	template<typename T>
	class vector
	{
	private:
		T* ptr;
		unsigned int size;
	public:
		__device__ vector() : size(0)
		{
			printf("Called vector()\n");
		}
		__device__ ~vector()
		{
			printf("Called ~vector()\n");
			if (ptr)
				delete[] ptr;
		}

		__device__ void push_back(const T& t)
		{
			printf("Called push_back()\n");
			unsigned int newSize = size + 1;
			T* newPtr = new T[newSize];
			for (unsigned int i = 0; i < size; i++)
			{
				printf("ptr[%d] is %d", i, ptr[i]);
				newPtr[i] = ptr[i];
			}
			newPtr[size] = t;
			size = newSize;
		}
		__device__ unsigned int Size()
		{
			printf("Called Size()\n");
			return size;
		}
	};
}

template <typename T>
class Foo
{
private:
	T val = 0;
public:
	__device__ Foo()
	{
		printf("Foo::Foo() is called\n");
	}
	__device__ virtual ~Foo()
	{
		printf("Foo::~Foo() is called\n");
	}

	__device__ virtual void show() noexcept
	{
		printf("%f",val);
	}
};

__global__ void kernel()
{
	std::vector<int> test;
	int size = test.Size();
	test.push_back(2);
	
}

int main()
{
	kernel << <1, 1>> > ();
}