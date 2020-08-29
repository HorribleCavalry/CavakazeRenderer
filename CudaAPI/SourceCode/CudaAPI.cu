#include "CudaAPI.cuh"
#include <array>
#include <vector>


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

__duel__ void testDuel()
{
	printf("Called testDuel");
}


class Base
{
public:
	Int a;
	__host__ __device__ Base()
	{
		a = 1;
	}

	__host__ __device__ virtual void print()
	{
		printf("Called Base!\n");
	}
};

class Child0 : public Base
{
public:
	Int b;
	Float c;
	__host__ __device__ Child0()
	{
		b = 1;
		c = 1.0;
	}

	__host__ __device__ virtual void print() override
	{
		printf("Called Child0!\n");
	}
};

class Child1 : public Base
{
public:
	Int d;
	Int e;
	__host__ __device__ Child1()
	{
		d = 1;
		e = 1;
	}
	__host__ __device__ virtual void print() override
	{
		printf("Called Child1!\n");
	}
};

__global__ void kernel()
{
	testDuel();
}

int main()
{
	//custd::cuvector<Base> vec;
	Base base;
	Base c0 = Child0();
	Base c1 = Child1();

	std::vector<Base> vec;
	vec.push_back(base);
	vec.push_back(c0);
	vec.push_back(c1);

	for each (Base it in vec)
	{
		it.print();
	}

	kernel << <1, 1>> > ();

}