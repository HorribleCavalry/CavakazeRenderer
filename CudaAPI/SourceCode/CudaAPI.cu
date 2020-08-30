#include "CudaAPI.cuh"
#include <array>
#include <vector>


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

__global__ void kernel()
{
	//vec2i i(0.0f);
	CUM::vec2i i;
	printf("%d\n", i.x);
	CUM::vec2f f;
	printf("%f\n", f.x);
	i = CUM::vec2i(2);
	printf("%d\n", i.x);
	f = CUM::vec2f(1.5);
	printf("%f\n", f.x);
	//i = f;
	//f = i;
	printf("%f\n", f.x);
	i = (CUM::vec2i)f;
	printf("%d\n", i.x);

	CUM::vec2i iAdded(6);
	printf("%d\n", iAdded.x);
	iAdded += i;
	printf("%d\n", iAdded.x);
	
	//f /= 0;

	//auto result = Add(i, f);

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
	CUM::vec2i i;
	CUM::vec2f f;
	i = (CUM::vec2i)f;
	kernel << <1, 1 >> > ();
}