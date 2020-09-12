#include "Common/Cuda3DMath.cuh"


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

__global__ void kernel()
{
	//CUM::vec2i vi0;
	//CUM::vec2i vi1;
	//CUM::vec2f vf0;
	//CUM::vec2f vf1;
	//vi0 = vi1;
	//vf0 = vi1;
	//vf1.x = 1.5;
	//vf0 = vf1;
	//vi0 = vf0;
	auto mat = CUM::Mat4x4_identity;
	CUM::Mat4x4i mati(5);
	mati += mat;
	CUM::Color3f color;
	CUM::calculateGammaColor(color, 2.2);
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

//__duel__ CUM::vec4<Float>&& reR()
//{
//	return CUM::vec4<Float>();
//}

int main()
{
	CUM::vec2i vi0;
	CUM::vec2i vi1(1.0f,2.0f);
	CUM::vec2f vf0;
	CUM::vec2f vf1(1.0f,2.0f);

	Int ni = 4;
	Float nf = 4.0;
	auto tempvi0 = vi0 + ni;
	auto tempvi1 = ni + vi0;
	auto tempvi2 = vi0 + nf;
	auto tempvi3 = nf + vi0;
	auto tempvi4 = vi0 + vi1;
	auto tempvi5 = vi0 + vf1;
	auto mat = CUM::Mat4x4_identity;
	CUM::Mat4x4i mati(5);
	mati += mat;
	mati -= mat;
	mati / mat;
	CUM::Mat4x4f matf;
	matf /= mat;
}