#include "Common/Cuda3DMath.cuh"
#include "Common/CudaPrimitivesVector.cuh"
#include "Common/Tools.cuh"
#include "Common/Geometry/Geometry.cuh"
#include "cuda/std/limits"
#include <chrono>

//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

class Person
{
public:
	__duel__ virtual void callType()
	{
		custd::OStream os;
		os << "I'm a person!" << custd::endl;
	}
};

class Student : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os<<"I'm a student!" << custd::endl;
	}
};

class Teacher : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a teacher!" << custd::endl;
	}
};

class Farmer : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a Farmer!" << custd::endl;
	}
};

class Heacker : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a Heacker!" << custd::endl;
	}
};

class Worker : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a Worker!" << custd::endl;
	}
};

__global__ void kernel()
{
	CUM::PrimitiveVector<Geometry> geoVec;
	Sphere sp;
	OBox bx;
	geoVec.push_back(sp);
	geoVec.push_back(bx);
	for (Int i = 0; i < geoVec.Size(); i++)
	{
		geoVec[i].GetArea();
	}
}

//__duel__ CUM::Vec4<Float>&& reR()
//{
//	return CUM::Vec4<Float>();
//}

class Base
{
public:
	Float a;
	__duel__ virtual void Call()
	{
		printf("Called Base::Call()\n");
	}
	Float GetStaticVariable()
	{
		static Float val = a / 2.0;
		return val;
	}
};

class Child : public Base
{
public:
	Int b;
	__duel__ virtual void Call() override
	{
		printf("Called Child::Call()\n");
	}

};


template<typename T>
__global__ void testCopiedInstance(T* ins)
{
	custd::OStream os;
	os << ins->sampleTime<<custd::endl;
}

template<typename T>
__global__ void testSceneCopy(T* ins)
{
	ins->Call();
	ins->camera->Call();
	auto vecPtr = ins->primitivesVectorPtr;
	auto vec = *vecPtr;
	Int size = vec.Size();
	for (Int i = 0; i < size; i++)
	{
		vec[i].Call();
	}
}

int main()
{

	PersCamera persCam;
	CUM::PrimitiveVector<Geometry> vec;
	Geometry geo;
	Sphere sp;
	BBox bb;
	OBox ob;
	Triangle tri;
	vec.push_back(geo);
	vec.push_back(sp);
	vec.push_back(bb);
	vec.push_back(ob);
	vec.push_back(tri);
	Scene scene(&persCam, &vec);
	Scene* sceneDevice = scene.copyToDevice();
	testSceneCopy << <1, 1 >> > (sceneDevice);

	Base b0;
	b0.a = 1.0;
	Base b1;
	b1.a = 2.0;
	custd::cout << b0.GetStaticVariable() << custd::endl;
	custd::cout << b1.GetStaticVariable() << custd::endl;
	Int width = 5;
	Int height = 5;

	CUM::Color3f* buffer = new CUM::Color3f[width*height];
	Int idx = height / 2 * width + height / 2;
	buffer[idx] = CUM::Color3f(1.0);
	Texture texture(CUM::Vec2i(width, height), buffer);
	CUM::Color3f result = texture.GetColor(CUM::Vec2f(0.5, 0.5));
	custd::cout << result.r << custd::endl;
	delete[] buffer;
}