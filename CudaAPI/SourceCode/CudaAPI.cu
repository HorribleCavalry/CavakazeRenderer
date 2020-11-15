#include "Common/Cuda3DMath.cuh"
#include "Common/CudaPrimitivesVector.cuh"
#include "Common/Tools.cuh"
#include "Common/Geometry/Geometry.cuh"
#include "cuda/std/limits"
#include "Common/stb_image.h"
#include "CudaAPI.cuh"
#include <chrono>
#include <string>
#include <thread>

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

__global__ void setup_kernel(curandState *state)
{
	Int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, globalIdx, 0, &state[globalIdx]);
}

__global__ void testTheRand(curandState *state)
{
	Int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;
	Int iteNum = globalIdx;
	curandState localState = state[globalIdx];
	printf("The %dth thread's rand num is: %f\nThe current blockIdx is: %d\nThe current threadIdx is: %d\n", globalIdx, curand_uniform(&localState), blockIdx.x, threadIdx.x);
}

struct cuvecTest
{
	Int idx;
	Bool* isRunning;
	cuvecTest(Int i, Bool* _isRunning) : idx(i), isRunning(_isRunning){}
	void operator()()
	{
		isRunning[idx] = true;
		custd::cuvector<Int> IntVec;
		for (Int i = 0; i < 10; i++)
		{
			IntVec.push_back(i);
			printf("The %dth thread printf is: %d\n", idx, IntVec[i]);
		}
		printf("Now release the %dth vec\n", idx);
		IntVec.Release();
		for (Int i = 0; i < 10; i++)
		{
			IntVec.push_back(i);
			printf("The %dth thread printf is: %d\n", idx, IntVec[i]);
		}
		isRunning[idx] = false;

	}
};

int main(int argc, char* argv[])
{
	//Bool* isRun = new Bool[2];
	//std::thread thread0(cuvecTest(0, isRun));
	//std::thread thread1(cuvecTest(1, isRun));

	//thread0.detach();
	//thread1.detach();
	//Bool isRunning = true;
	//while (isRunning)
	//{
	//	isRunning = true;
	//	isRunning = isRunning && isRun[0];
	//	for (Int i = 0; i < 2; i++)
	//	{
	//		isRunning = isRunning || isRun[i];
	//	}
	//}
	//delete[] isRun;
	std::string exePath = argv[0];//获取当前程序所在的路径
	std::string hierarchyPath = exePath.substr(0, exePath.find_last_of("\\") + 1);
	const char* imageName = "Image.ppm";
	std::string imagePath = hierarchyPath + imageName;

	Int width = 2560;
	Int height = 1440;
	const Int bounceTime = 64;
	const Int ranNumSize = 2048;

	CUM::Vec2i RenderTargetSize(width, height);
	Int imageLength = RenderTargetSize.x * RenderTargetSize.y;
#ifdef RUN_ON_DEVICE
	InitDeviceStates(imageLength);
#endif // RUN__ON__DEVICE

	Pixel* buffer = new Pixel[imageLength];
	Texture* RenderTarget = new Texture(RenderTargetSize, buffer);
	PersCamera* camera = new PersCamera(CUM::Point3f(0.0, 0.0, 0.0), { 0.0,0.0,1.0 }, CUM::Quaternionf({ 0.0,1.0,0.0 }, 0.0, true), RenderTargetSize, 0.1, 10000.0, bounceTime, 0.5 * PI, RenderTarget);

	//Int spherePerDimNum = 10;
	//Float sphereRadius = 0.25;
	//Float inValPerNum = 1.0 / spherePerDimNum;
	//Float incTransPerNum = 5.0 / spherePerDimNum;
	//for (Int i = 0; i < spherePerDimNum; i++)
	//{
	//	for (Int i = 0; i < spherePerDimNum; i++)
	//	{

	//	}
	//}

	//Geometry* sp0 = new Sphere(CUM::Point3f(-2.5, -1.65, 5.0), 0.25);
	//Geometry* sp1 = new Sphere(CUM::Point3f(2.5, -1.65, 5.0), 0.25);

	//CUM::PrimitiveVector<Geometry>* primitiveVec0 = new CUM::PrimitiveVector<Geometry>;
	//primitiveVec0->push_back(*sp0);
	//primitiveVec0->push_back(*sp1);

	Geometry* box0 = new BBox(CUM::Point3f(0.0, -2.1, 7.5), CUM::Vec3f(5.2, 0.1, 5.2));
	CUM::PrimitiveVector<Geometry>* primitiveVec1 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec1->push_back(*box0);

	Geometry* box1 = new BBox(CUM::Point3f(-5.1, 0.0, 7.5), CUM::Vec3f(0.1, 2.0, 5.0));
	CUM::PrimitiveVector<Geometry>* primitiveVec2 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec2->push_back(*box1);

	Geometry* box2 = new BBox(CUM::Point3f(5.1, 0.0, 7.5), CUM::Vec3f(0.1, 2.0, 5.0));
	CUM::PrimitiveVector<Geometry>* primitiveVec3 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec3->push_back(*box2);

	Geometry* box3 = new BBox(CUM::Point3f(0.0, 2.1, 7.5), CUM::Vec3f(5.2, 0.1, 5.2));
	CUM::PrimitiveVector<Geometry>* primitiveVec4 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec4->push_back(*box3);

	Geometry* box4 = new BBox(CUM::Point3f(0.0, 0.0, 12.6), CUM::Vec3f(5.2, 2.0, 0.1));
	CUM::PrimitiveVector<Geometry>* primitiveVec5 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec5->push_back(*box4);

	//Material* material0 = new Material;
	//material0->metallic = 0.0;
	//material0->Albedo = CUM::Color3f(0.4, 0.8, 0.8);
	//material0->roughness = 0.25;

	Material* material1 = new Lambert;
	material1->roughness = 1.0;
	material1->metallic = 0.0;
	material1->Albedo = CUM::Color3f(0.85, 0.85, 0.85);

	Material* material2 = new Lambert;
	material2->roughness = 1.0;
	material2->metallic = 0.0;
	material2->Albedo = CUM::Color3f(1.0, 0.5, 0.5);

	Material* material3 = new Lambert;
	material3->roughness = 1.0;
	material3->metallic = 0.0;
	material3->Albedo = CUM::Color3f(0.5, 1.0, 0.5);

	Material* material4 = new Lambert;
	material4->roughness = 1.0;
	material4->metallic = 0.0;
	material4->Albedo = CUM::Color3f(0.5, 0.5, 1.0);

	Material* material5 = new Lambert;
	material5->roughness = 1.0;
	material5->metallic = 0.0;
	material5->Albedo = CUM::Color3f(0.85, 0.85, 0.85);

	//Mesh* mesh0 = new Mesh(primitiveVec0, material0);
	Mesh* mesh1 = new Mesh(primitiveVec1, material1);
	Mesh* mesh2 = new Mesh(primitiveVec2, material2);
	Mesh* mesh3 = new Mesh(primitiveVec3, material3);
	Mesh* mesh4 = new Mesh(primitiveVec4, material4);
	Mesh* mesh5 = new Mesh(primitiveVec5, material5);

	CUM::PrimitiveVector<Mesh>* meshVec0 = new CUM::PrimitiveVector<Mesh>;
	//meshVec0->push_back(*mesh0);
	meshVec0->push_back(*mesh1);
	meshVec0->push_back(*mesh2);
	meshVec0->push_back(*mesh3);
	meshVec0->push_back(*mesh4);
	meshVec0->push_back(*mesh5);

	CUM::Vec3f scale(1.0, 1.0, 1.0);
	CUM::Vec3f translation(0.0, 0.0, 2.0);
	CUM::Quaternionf rotation(CUM::Vec3f(0.0, 1.0, 0.0), 0.0);
	CUM::Transform trans(scale, rotation, translation);

	Object* object = new Object(trans, meshVec0);

	CUM::PrimitiveVector<Object>* objectVec = new CUM::PrimitiveVector<Object>;
	Float W = 10;

	for (Int i = 0; i < 5; i++)
	{
		for (Int j = 0; j < 5; j++)
		{
			Geometry* spR = new Sphere(CUM::Point3f(2.0* i + 0.25 - 5.0, -1.65, 2.0 * j + 0.25 + 0.5 * 7.5), 0.25);
			CUM::PrimitiveVector<Geometry>* geoVec = new CUM::PrimitiveVector<Geometry>;
			geoVec->push_back(*spR);
			Material* mat = new Material;
			mat->Albedo = CUM::Color3f(0.4, 0.8, 0.8);
			mat->roughness = i / 5.0;
			mat->metallic = j / 5.0;
			Mesh* mesh = new Mesh(geoVec, mat);
			meshVec0->push_back(*mesh);
		}
	}

	objectVec->push_back(*object);

	Scene scene(camera, objectVec);
	Scene* sceneDevice = scene.copyToDevice();
	//TestMaterialRandVec << <1, 1 >> > (sceneDevice);

	auto start = std::chrono::steady_clock::now();
	Rendering(&scene, sceneDevice, imageLength);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> duration = end - start;
	custd::cout << duration.count() << custd::endl;

	scene.camera->renderTarget->Save(imagePath.c_str());
	custd::cout << "Now release host scene." << custd::endl;
	scene.Release();
	custd::cout << "Now release device scene." << custd::endl;
	ReleaseIns << <1, 1 >> > (sceneDevice);
}

CUDA_API void OpenDebugConsole()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
}

CUDA_API void CloseDebugConsole()
{
	FreeConsole();
}

CUDA_API void StartRendering(Int width, Int height, void* imagePtr)
{
	std::string exePath = "D:\GitSpace\CavakazeRenderer\Release\CavakazeRenderer.exe";
	std::string hierarchyPath = exePath.substr(0, exePath.find_last_of("\\") + 1);
	const char* imageName = "Image.ppm";
	std::string imagePath = hierarchyPath + imageName;

	const Int bounceTime = 64;
	const Int ranNumSize = 2048;

	CUM::Vec2i RenderTargetSize(width, height);
	Int imageLength = RenderTargetSize.x * RenderTargetSize.y;
#ifdef RUN_ON_DEVICE
	InitDeviceStates(imageLength);
#endif // RUN__ON__DEVICE

	//Pixel* buffer = new Pixel[imageLength];
	Pixel* buffer = (Pixel*)imagePtr;
	Texture* RenderTarget = new Texture(RenderTargetSize, buffer);
	PersCamera* camera = new PersCamera(CUM::Point3f(0.0, 0.0, 0.0), { 0.0,0.0,1.0 }, CUM::Quaternionf({ 0.0,1.0,0.0 }, 0.0, true), RenderTargetSize, 0.1, 10000.0, bounceTime, 0.5 * PI, RenderTarget);

	//Int spherePerDimNum = 10;
	//Float sphereRadius = 0.25;
	//Float inValPerNum = 1.0 / spherePerDimNum;
	//Float incTransPerNum = 5.0 / spherePerDimNum;
	//for (Int i = 0; i < spherePerDimNum; i++)
	//{
	//	for (Int i = 0; i < spherePerDimNum; i++)
	//	{

	//	}
	//}

	//Geometry* sp0 = new Sphere(CUM::Point3f(-2.5, -1.65, 5.0), 0.25);
	//Geometry* sp1 = new Sphere(CUM::Point3f(2.5, -1.65, 5.0), 0.25);

	//CUM::PrimitiveVector<Geometry>* primitiveVec0 = new CUM::PrimitiveVector<Geometry>;
	//primitiveVec0->push_back(*sp0);
	//primitiveVec0->push_back(*sp1);

	Geometry* box0 = new BBox(CUM::Point3f(0.0, -2.1, 7.5), CUM::Vec3f(5.2, 0.1, 5.2));
	CUM::PrimitiveVector<Geometry>* primitiveVec1 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec1->push_back(*box0);

	Geometry* box1 = new BBox(CUM::Point3f(-5.1, 0.0, 7.5), CUM::Vec3f(0.1, 2.0, 5.0));
	CUM::PrimitiveVector<Geometry>* primitiveVec2 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec2->push_back(*box1);

	Geometry* box2 = new BBox(CUM::Point3f(5.1, 0.0, 7.5), CUM::Vec3f(0.1, 2.0, 5.0));
	CUM::PrimitiveVector<Geometry>* primitiveVec3 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec3->push_back(*box2);

	Geometry* box3 = new BBox(CUM::Point3f(0.0, 2.1, 7.5), CUM::Vec3f(5.2, 0.1, 5.2));
	CUM::PrimitiveVector<Geometry>* primitiveVec4 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec4->push_back(*box3);

	Geometry* box4 = new BBox(CUM::Point3f(0.0, 0.0, 12.6), CUM::Vec3f(5.2, 2.0, 0.1));
	CUM::PrimitiveVector<Geometry>* primitiveVec5 = new CUM::PrimitiveVector<Geometry>;
	primitiveVec5->push_back(*box4);

	//Material* material0 = new Material;
	//material0->metallic = 0.0;
	//material0->Albedo = CUM::Color3f(0.4, 0.8, 0.8);
	//material0->roughness = 0.25;

	Material* material1 = new Lambert;
	material1->roughness = 1.0;
	material1->metallic = 0.0;
	material1->Albedo = CUM::Color3f(0.85, 0.85, 0.85);

	Material* material2 = new Lambert;
	material2->roughness = 1.0;
	material2->metallic = 0.0;
	material2->Albedo = CUM::Color3f(1.0, 0.5, 0.5);

	Material* material3 = new Lambert;
	material3->roughness = 1.0;
	material3->metallic = 0.0;
	material3->Albedo = CUM::Color3f(0.5, 1.0, 0.5);

	Material* material4 = new Lambert;
	material4->roughness = 1.0;
	material4->metallic = 0.0;
	material4->Albedo = CUM::Color3f(0.5, 0.5, 1.0);

	Material* material5 = new Lambert;
	material5->roughness = 1.0;
	material5->metallic = 0.0;
	material5->Albedo = CUM::Color3f(0.85, 0.85, 0.85);

	//Mesh* mesh0 = new Mesh(primitiveVec0, material0);
	Mesh* mesh1 = new Mesh(primitiveVec1, material1);
	Mesh* mesh2 = new Mesh(primitiveVec2, material2);
	Mesh* mesh3 = new Mesh(primitiveVec3, material3);
	Mesh* mesh4 = new Mesh(primitiveVec4, material4);
	Mesh* mesh5 = new Mesh(primitiveVec5, material5);

	CUM::PrimitiveVector<Mesh>* meshVec0 = new CUM::PrimitiveVector<Mesh>;
	//meshVec0->push_back(*mesh0);
	meshVec0->push_back(*mesh1);
	meshVec0->push_back(*mesh2);
	meshVec0->push_back(*mesh3);
	meshVec0->push_back(*mesh4);
	meshVec0->push_back(*mesh5);

	CUM::Vec3f scale(1.0, 1.0, 1.0);
	CUM::Vec3f translation(0.0, 0.0, 2.0);
	CUM::Quaternionf rotation(CUM::Vec3f(0.0, 1.0, 0.0), 0.0);
	CUM::Transform trans(scale, rotation, translation);

	Object* object = new Object(trans, meshVec0);

	CUM::PrimitiveVector<Object>* objectVec = new CUM::PrimitiveVector<Object>;
	Float W = 10;

	for (Int i = 0; i < 5; i++)
	{
		for (Int j = 0; j < 5; j++)
		{
			Geometry* spR = new Sphere(CUM::Point3f(2.0* i + 0.25 - 5.0, -1.65, 2.0 * j + 0.25 + 0.5 * 7.5), 0.25);
			CUM::PrimitiveVector<Geometry>* geoVec = new CUM::PrimitiveVector<Geometry>;
			geoVec->push_back(*spR);
			Material* mat = new Material;
			mat->Albedo = CUM::Color3f(0.4, 0.8, 0.8);
			mat->roughness = i / 5.0;
			mat->metallic = j / 5.0;
			Mesh* mesh = new Mesh(geoVec, mat);
			meshVec0->push_back(*mesh);
		}
	}

	objectVec->push_back(*object);

	Scene scene(camera, objectVec);
	Scene* sceneDevice;
#ifdef RUN_ON_DEVICE
	sceneDevice = scene.copyToDevice();
#endif // RUN__ON__DEVICE

	auto start = std::chrono::steady_clock::now();
	Rendering(&scene, sceneDevice, imageLength);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> duration = end - start;
	custd::cout << duration.count() << custd::endl;

	//scene.camera->renderTarget->Save(imagePath.c_str());
	custd::cout << "Now release host scene." << custd::endl;
	scene.Release();
#ifdef RUN_ON_DEVICE
	custd::cout << "Now release device scene." << custd::endl;
	ReleaseIns << <1, 1 >> > (sceneDevice);
#endif // RUN__ON__DEVICE
}