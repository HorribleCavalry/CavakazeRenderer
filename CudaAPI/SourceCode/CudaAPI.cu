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
	std::string exePath = argv[0];//获取当前程序所在的路径
	std::string hierarchyPath = exePath.substr(0, exePath.find_last_of("\\") + 1);
	const char* imageName = "Image.ppm";
	std::string imagePath = hierarchyPath + imageName;

	Int width = 256;
	Int height = 144;
	CUM::Vec2i RenderTargetSize(width, height);
	Int imageLength = RenderTargetSize.x * RenderTargetSize.y;
	Pixel* buffer = new Pixel[imageLength];
	Texture* RenderTarget = new Texture(RenderTargetSize, buffer);
	PersCamera* camera = new PersCamera({ 0.0 }, { 0.0,1.0,0.0 }, CUM::Quaternionf({ 0.0,1.0,0.0 }, 0.0, true), RenderTargetSize, 0.1, 10000.0, 1, 0.5 * PI, RenderTarget);
	//Sphere* sps = new Sphere[3];
	//sps[0].centroid = CUM::Point3f(-5.0, 0.0, 10.0);
	//sps[1].centroid = CUM::Point3f(0.0, 0.0, 10.0);
	//sps[2].centroid = CUM::Point3f(5.0, 0.0, 10.0);

	Geometry* sp0 = new Sphere(CUM::Point3f(-5.0, 0.0, 10.0), 1.0);
	Geometry* box0 = new BBox(CUM::Point3f(0.0, 0.0, 10.0), CUM::Vec3f(1.0));
	Geometry* sp1 = new Sphere(CUM::Point3f(5.0, 0.0, 10.0), 1.0);

	CUM::PrimitiveVector<Geometry>* primitiveVec = new CUM::PrimitiveVector<Geometry>;
	//for (Int i = 0; i < 3; i++)
	//{
	//	primitiveVec->push_back(sps[i]);
	//}

	primitiveVec->push_back(*sp0);
	primitiveVec->push_back(*box0);
	primitiveVec->push_back(*sp1);

	Scene scene(camera, primitiveVec);
	Scene* sceneDevice = scene.copyToDevice();

	Int threadNum = 32;
	Int blockNum = imageLength / threadNum;

	//RenderingOnDevice << < blockNum, threadNum >> > (sceneDevice);
	RenderingOnHost(&scene);
	cudaError_t error = cudaGetLastError();

	if (error != cudaError_t::cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	//scene.camera->renderTarget->CopyFromDevice(PersCamera::RenderTargetDevice);
	scene.camera->renderTarget->Save(imagePath.c_str());
	custd::cout << "Now release host scene." << custd::endl;
	scene.Release();
	//custd::cout << "Now release device scene." << custd::endl;
	//ReleaseIns << <1, 1 >> > (sceneDevice);

	//CUM::Vec2i imageSize(1920, 1080);
	//Int imageLength = imageSize.x*imageSize.y;
	//CUM::Color3f* uvBuffer = new CUM::Color3f[imageLength];
	//Texture* renderTarget = new Texture(imageSize, uvBuffer);

	//std::string exePath = argv[0];//获取当前程序所在的路径
	//std::string hierarchyPath = exePath.substr(0, exePath.find_last_of("\\") + 1);
	//const char* imageName = "Image.ppm";
	//std::string imagePath = hierarchyPath + imageName;

	//Texture* renderTargetDevice = renderTarget->copyToDevice();
	//Int threadNum = 1024;
	//Int blockNum = imageLength / threadNum;
	//renderUV << < blockNum, threadNum >> > (renderTargetDevice);
	//renderTarget->CopyFromDevice(renderTargetDevice);
	//renderTarget->Save(imagePath.c_str());

}