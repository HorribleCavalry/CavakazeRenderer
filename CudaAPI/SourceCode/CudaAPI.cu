#include "Common/Cuda3DMath.cuh"
#include "Common/CudaPrimitivesVector.cuh"
#include "Common/Tools.cuh"
#include "Common/Geometry/Geometry.cuh"
#include "cuda/std/limits"
#include "Common/stb_image.h"
#include <chrono>
#include <string>
#include <random>


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
__global__ void ReleaseIns(T* ins)
{
	ins->Release();
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

	custd::OStream os;
	os << ins->camera->renderTarget->GetColor(CUM::Vec2f(0.5, 0.5)).r << custd::endl;
}

__global__ void renderUV(Texture* renderTargetDevice)
{
	Int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;

	Float u = Float(globalIdx % renderTargetDevice->width) / Float(renderTargetDevice->width);
	Float v = Float(globalIdx / renderTargetDevice->width) / Float(renderTargetDevice->height);
	renderTargetDevice->buffer[globalIdx].r = u;
	renderTargetDevice->buffer[globalIdx].g = v;
}

__duel__ void outputStatic(Int globalIdx)
{
	static Int calledTime = 0;
	++calledTime;
	printf("Now called the globalIdx is %d, the calledTime is %d.\n", globalIdx, calledTime);
}

__global__ void callStaticTest()
{
	Int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;

	outputStatic(globalIdx);
	outputStatic(globalIdx);
}

__global__ void TestMaterialRandVec(Scene* scene)
{
	//printf("Now testing material copy rand vec on device!\n");
	(*(*scene->objectVec)[0].meshVec)[0].material->testForCopyRandVec();
	(*(*scene->objectVec)[0].meshVec)[1].material->testForCopyRandVec();
	//printf("Now end testing material copy rand vec on device!\n");
}


int main(int argc, char* argv[])
{
	std::string exePath = argv[0];//获取当前程序所在的路径
	std::string hierarchyPath = exePath.substr(0, exePath.find_last_of("\\") + 1);
	const char* imageName = "Image.ppm";
	std::string imagePath = hierarchyPath + imageName;

	Int width = 2560;
	Int height = 1440;
	const Int bounceTime = 64;
	const Int ranNumSize = 2048;
	//const Int aliasingTime = 16;

	CUM::Vec2i RenderTargetSize(width, height);
	Int imageLength = RenderTargetSize.x * RenderTargetSize.y;
	Pixel* buffer = new Pixel[imageLength];
	Texture* RenderTarget = new Texture(RenderTargetSize, buffer);
	PersCamera* camera = new PersCamera(CUM::Point3f(0.0, 0.0, 0.0), { 0.0,0.0,1.0 }, CUM::Quaternionf({ 0.0,1.0,0.0 }, 0.0, true), RenderTargetSize, 0.1, 10000.0, bounceTime, 0.5 * PI, RenderTarget);
	

	//Geometry* sp0 = new Sphere(CUM::Point3f(0.0, 0.0, 10.0), 1.0);
	Geometry* sp0 = new Sphere(CUM::Point3f(-5.0, 0.0, 10.0), 1.0);
	Geometry* sp1 = new Sphere(CUM::Point3f(5.0, 0.0, 10.0), 1.0);
	//Geometry* box0 = new BBox(CUM::Point3f(0.0, 0.0, 10.0), CUM::Vec3f(1.0));

	CUM::PrimitiveVector<Geometry>* primitiveVec0 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec0->push_back(*sp0);
	primitiveVec0->push_back(*sp1);

	Geometry* box1 = new BBox(CUM::Point3f(0.0, -1.11, 10.0), CUM::Vec3f(15.0, 0.1, 15.0));

	CUM::PrimitiveVector<Geometry>* primitiveVec1 = new CUM::PrimitiveVector<Geometry>;

	primitiveVec1->push_back(*box1);

	Material* material0 = new Material;
	material0->Albedo = CUM::Color3f(0.4, 0.8, 0.8);
	material0->InitializeRandVecs();

	Material* material1 = new Material;
	material1->Albedo = CUM::Color3f(0.85, 0.85, 0.85);
	material1->InitializeRandVecs();

	Mesh* mesh0 = new Mesh(primitiveVec0,material0);
	Mesh* mesh1 = new Mesh(primitiveVec1, material1);
	CUM::PrimitiveVector<Mesh>* meshVec0 = new CUM::PrimitiveVector<Mesh>;
	meshVec0->push_back(*mesh0);
	meshVec0->push_back(*mesh1);

	CUM::Vec3f scale(1.0, 1.0, 1.0);
	CUM::Vec3f translation(0.0, 0.0, 0.0);
	CUM::Quaternionf rotation(CUM::Vec3f(0.0, 1.0, 0.0), 0.25 * PI);
	CUM::Transform trans(scale, rotation, translation);

	Object* object = new Object(trans, meshVec0);

	CUM::PrimitiveVector<Object>* objectVec = new CUM::PrimitiveVector<Object>;

	objectVec->push_back(*object);

	Scene scene(camera, objectVec);
	Scene* sceneDevice = scene.copyToDevice();
	//TestMaterialRandVec << <1, 1 >> > (sceneDevice);

	Bool isOnDevice = false;
	Rendering(&scene, sceneDevice, imageLength, isOnDevice);


	scene.camera->renderTarget->Save(imagePath.c_str());
	custd::cout << "Now release host scene." << custd::endl;
	scene.Release();
	custd::cout << "Now release device scene." << custd::endl;
	ReleaseIns << <1, 1 >> > (sceneDevice);
}