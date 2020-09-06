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
	CUM::vec4<Float> v4l;
	CUM::vec4<Float>&& v4r = CUM::vec4<Float>(0, 0, 0, 1);
	CUM::vec4<Float> v4lt(v4l);
	CUM::vec4<Float> v4rt(v4r);
	//f /= 0;
	printf("%f\n",CUM::Mat4x4_identity.m[1][1]);

	//auto result = Add(i, f);
	//printf("%d\n", creation::human.id);
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

__duel__ CUM::vec4<Float>&& reR()
{
	return CUM::vec4<Float>();
}

int main()
{
	CUM::vec2i i;
	CUM::vec2f f;
	i = (CUM::vec2i)f;
	CUM::vec4<Float> vL;
	CUM::vec4<Float> v(reR());
	kernel << <1, 1 >> > ();
	//printf("%d\n", creation::human.human.id);
}