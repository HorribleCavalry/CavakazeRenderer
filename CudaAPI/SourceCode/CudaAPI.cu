#include "CudaAPI.cuh"


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

__global__ void kernel()
{
	CUM::vec2i vi0;
	CUM::vec2i vi1;
	CUM::vec2f vf0;
	CUM::vec2f vf1;
	//vi0 = vi1;
	vf0 = vi1;
	vf1.x = 1.5;
	vf0 = vf1;
	//vi0 = vf0;
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
	CUM::vec2i vi0;
	CUM::vec2i vi1;
	CUM::vec2f vf0;
	CUM::vec2f vf1;
	//vi0 = vi1;
	vf0 = vi1;
	vf1.x = 1.5;
	vf0 = vf1;
	//vi0 = (CUM::vec2i)vf0;
	//vi0 = vf0;
	vf0 = vi0;
	//vf1 = vf1 + vf1;
	//vf1 = vi0 + vf1;
	//vf0 = 1.0 + vi0;
}