#include "Common/Cuda3DMath.cuh"
#include "Common/CudaPrimitivesVector.cuh"
#include "Common/Tools.cuh"
#include "Common/Geometry/Geometry.cuh"
#include "cuda/std/limits"
#include "Common/stb_image.h"
#include <chrono>
#include <string>


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
	//custd::OStream os;
	//os << ins->sampleTime<<custd::endl;
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

int main(int argc, char* argv[])
{
	//Int width = 5;
	//Int height = 5;
	//CUM::Color3f* buffer = new CUM::Color3f[width*height];
	//Int idx = height / 2 * width + height / 2;
	//buffer[idx] = CUM::Color3f(1.0);
	//Texture* RenderTarget = new Texture(CUM::Vec2i(width, height), buffer);
	//PersCamera* persCam = new PersCamera;
	//persCam->renderTarget = RenderTarget;
	//CUM::PrimitiveVector<Geometry>* vec = new CUM::PrimitiveVector<Geometry>;
	//Geometry* geo = new Geometry;
	//Sphere* sp = new Sphere;
	//BBox* bb = new BBox;
	//OBox* ob = new OBox;
	//Triangle* tri = new Triangle;
	//vec->push_back(*geo);
	//vec->push_back(*sp);
	//vec->push_back(*bb);
	//vec->push_back(*ob);
	//vec->push_back(*tri);
	//Scene scene(persCam, vec);
	//Scene* sceneDevice = scene.copyToDevice();
	//scene.Release();
	//ReleaseIns << <1, 1 >> > (sceneDevice);



	CUM::Vec2i imageSize(1920, 1080);
	Int imageLength = imageSize.x*imageSize.y;
	CUM::Color3f* uvBuffer = new CUM::Color3f[imageLength];
	Texture* renderTarget = new Texture(imageSize, uvBuffer);

	std::string exePath = argv[0];//获取当前程序所在的路径
	std::string hierarchyPath = exePath.substr(0, exePath.find_last_of("\\") + 1);
	const char* imageName = "Image.ppm";
	std::string imagePath = hierarchyPath + imageName;

	Texture* renderTargetDevice = renderTarget->copyToDevice();
	Int threadNum = 1024;
	Int blockNum = imageLength / threadNum;
	renderUV << < blockNum, threadNum >> > (renderTargetDevice);
	renderTarget->CopyFromDevice(renderTargetDevice);
	renderTarget->Save(imagePath.c_str());

}