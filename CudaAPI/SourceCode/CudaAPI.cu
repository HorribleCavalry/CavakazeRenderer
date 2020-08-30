#include "CudaAPI.cuh"
#include <array>
#include <vector>


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

__global__ void kernel()
{
	//vec2i i(0.0f);
	vec2i i;
	printf("%d\n", i.x);
	vec2f f;
	printf("%f\n", f.x);
	i = vec2i(2);
	printf("%d\n", i.x);
	f = vec2f(1.5);
	printf("%f\n", f.x);
	//i = f;
	//f = i;
	printf("%f\n", f.x);
	i = (vec2i)f;
	printf("%d\n", i.x);

	auto result = Add(i, f);

}


//class Person
//{
//public:
//	Person() {}
//	Person(const Person&) {}
//	Person(Person&&) {}
//	const Person& operator=(const Person&) {}
//	const Person& operator=(Person&&) {}
//	~Person() {}
//};

int main()
{
	vec2i i;
	vec2f f;
	i = (vec2i)f;
	kernel << <1, 1 >> > ();
}