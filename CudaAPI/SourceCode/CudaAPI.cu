#include "Common/Cuda3DMath.cuh"
#include "Common/CudaPrimitivesVector.cuh"
#include "Common/Tools.cuh"
#include "Common/Geometry/Geometry.cuh"
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


__global__ void testVirtualBetweenHostAndDevice(Camera* cam)
{
	cam->Call();
}

int main()
{
	//Person** prList = new Person*[5];

	//Person per0;
	//Student stu;
	//Farmer far;
	//Heacker hea;
	//Worker wor;
	//prList[0] = &per0;
	//prList[1] = &stu;
	//prList[2] = &far;
	//prList[3] = &hea;
	//prList[4] = &wor;
	//
	//for (int i = 0; i < 5; i++)
	//{
	//	prList[i]->callType();
	//}

	//CUM::PrimitiveVector<Person> list;
	//list.push_back(per0);
	//list.push_back(stu);
	//list.push_back(far);
	//list.push_back(hea);
	//list.push_back(wor);
	//for (Int i = 0; i < 5; i++)
	//{
	//	list[i].callType();
	//}

	CUM::Vec2i vi0;
	CUM::Vec2i vi1(1.0f, 2.0f);
	CUM::Vec2f vf0;
	CUM::Vec2f vf1(1.0f, 2.0f);
	CUM::Vec4f vf4(2.0f);
	auto test =CUM::normalize(vf4);
	CUM::Vec3f vf41(2.0f);
	auto test1 = CUM::normalize(vf41);

	Person per0;
	Student stu;
	Farmer far;
	Heacker hea;
	Worker wor;

	CUM::PrimitiveVector<Person> list;
	list.push_back(per0);
	list.push_back(stu);
	list.push_back(far);
	list.push_back(hea);
	list.push_back(wor);
	for (Int i = 0; i < 5; i++)
	{
		list[i].callType();
	}

	CUM::Quaternion<Float> rotatePositive(CUM::Vec3f(0, 1, 0), 0.25*PI, true);
	CUM::Vec4f trans(1.0, 1.0, 1.0, 0.0);
	CUM::Vec4f origin(1.0, 1.0, 1.0, 0.0);
	
	auto start = std::chrono::steady_clock::now();

	for (Int i = 0; i < 1048576; i++)
	{
		auto temp = CUM::applyQuaTransform(rotatePositive, origin);
		temp *= 2.0;
		temp /= 2.0;
		temp += trans;
		temp -= trans;
	}
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<Float> duration = end - start;
	//0.41030820000000001 second

	start = std::chrono::steady_clock::now();
	kernel << <1, 1 >> > ();
	end = std::chrono::steady_clock::now();
	duration = end - start;
	//0.41093520000000000 second

	Ray r;
	CUM::PrimitiveVector<Geometry> geoVec;
	Sphere sp;
	OBox bx;
	geoVec.push_back(sp);
	geoVec.push_back(bx);
	for (Int i = 0; i < geoVec.Size(); i++)
	{
		geoVec[i].GetArea();
	}

	CUM::Point3f p0;
	CUM::Vec3f vfff;
	p0 = vfff;
	//Geometry g;

	Scene scene;
	//Sphere sp;
	scene.AddPrimitive(sp);
	Camera* perCamHost = new Camera;
	Camera* perCamDevice;
	cudaMalloc(&perCamDevice, sizeof(Camera));
	cudaMemcpy(perCamDevice, perCamHost, sizeof(Camera), cudaMemcpyKind::cudaMemcpyHostToDevice);
	testVirtualBetweenHostAndDevice << <1, 1 >> > (perCamDevice);
	//perCamHost->Call();
	//scene.Rendering();
}