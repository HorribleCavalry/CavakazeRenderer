﻿#ifndef __TOOLS__CUH__
#define __TOOLS__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "Cuda3DMath.cuh"
#include <random>
#include <fstream>
#include <iomanip>

class Material
{
public:
	Float metallic = 0.0;
	Float roughness = 0.5;
	Float F0 = 0.05;
	CUM::Color3f Albedo;
public:
	Bool isHemisphere = false;
public:
	CUM::Vec3f* randVecs;
	Float* randNums;
	Int interactTime;
	Int randSize;
public:
	__duel__ Material()
	{
		interactTime = 0;
		randSize = 64;
	}
	__host__ virtual void InitializeRandVecs()
	{
		CHECK(randSize > 0, "Material::InitializeRandVecs() error: the randVecSize can not less than 0!");
		randVecs = new CUM::Vec3f[randSize];
		randNums = new Float[randSize];

		std::default_random_engine randEngine(rand());
		std::uniform_real_distribution<Float> randGenerator(0.0, 1.0);
		Float xi1, xi2;
		Float x, y, z;
		Float Sqrt1MinusXi1Square;
		for (Int i = 0; i < randSize; i++)
		{
			randNums[i] = randGenerator(randEngine);

			xi1 = randGenerator(randEngine);
			xi2 = randGenerator(randEngine);
			Sqrt1MinusXi1Square = sqrt(1.0 - xi1 * xi1);
			x = cos(2.0*PI*xi2)*Sqrt1MinusXi1Square;
			z = sin(2.0*PI*xi2)*Sqrt1MinusXi1Square;
			y = xi1;
			randVecs[i] = CUM::Vec3f(x, y, z);
		}

	}
public:
	__host__ Material* copyToDevice()
	{
		Material materialInsWithDevicePtr(*this);
		CUM::Vec3f* randVecsDevice;
		Float* randNumsDevice;
		
		cudaMalloc(&randVecsDevice, randSize * sizeof(CUM::Vec3f));
		cudaMemcpy(randVecsDevice, randVecs, randSize * sizeof(CUM::Vec3f), cudaMemcpyKind::cudaMemcpyHostToDevice);

		cudaMalloc(&randNumsDevice, randSize * sizeof(Float));
		cudaMemcpy(randNumsDevice, randNums, randSize * sizeof(Float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		materialInsWithDevicePtr.randVecs = randVecsDevice;
		materialInsWithDevicePtr.randNums = randNumsDevice;

		Material* materialDevice = CudaInsMemCpyHostToDevice(&materialInsWithDevicePtr);
		return materialDevice;
	}
	__duel__ void Release()
	{
		if (randVecs)
		{
			delete[] randVecs;
			randVecs = nullptr;
		}

		if (randNums)
		{
			delete[] randNums;
			randNums = nullptr;
		}
	}

#ifdef RUN_ON_DEVICE
__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
__host__
#endif // RUN_ON_HOST
	const Float DistributionGGX(const CUM::Vec3f& N, const CUM::Vec3f& H, const Float& roughness)
	{
		Float a = roughness * roughness;
		Float a2 = a * a;
		Float NdotH = CUM::max(CUM::dot(N, H), 0.0);
		Float NdotH2 = NdotH * NdotH;

		Float nom = a2;
		Float denom = NdotH2*(a2 - 1.0) + 1.0;
		denom = PI * denom*denom;
		return nom / denom;
	}
#ifdef RUN_ON_DEVICE
__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
__host__
#endif // RUN_ON_HOST
	const Float GeometrySchlickGGX(Float NdotV, Float roughness)
	{
		Float r = (roughness + 1.0);
		Float k = (r*r) / 8.0;
	
		Float nom = NdotV;
		Float denom = NdotV * (1.0 - k) + k;
	
		return nom / denom;
	}

#ifdef RUN_ON_DEVICE
__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
__host__
#endif // RUN_ON_HOST
	const Float GeometrySmith(const CUM::Vec3f& N, const CUM::Vec3f& V, const CUM::Vec3f& L, Float roughness)
	{
		Float NdotV = CUM::max(CUM::dot(N, V), 0.0);
		Float NdotL = CUM::max(CUM::dot(N, L), 0.0);
		Float ggx2 = GeometrySchlickGGX(NdotV, roughness);
		Float ggx1 = GeometrySchlickGGX(NdotL, roughness);
	
		return ggx1 * ggx2;
	}

#ifdef RUN_ON_DEVICE
__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
__host__
#endif // RUN_ON_HOST
	virtual const CUM::Color3f ShadeWithDirectLight(const CUM::Vec3f& N, const CUM::Vec3f& V, const CUM::Vec3f& H, const CUM::Vec3f& L, const Float& lightDis,const CUM::Color3f& LightRadiance)
	{
		CUM::Color3f Lo(0.0);

		Float attenuation = CUM::min(1.0 / (lightDis*lightDis), 1.0);

		Float NdotH = CUM::dot(N, H);
		Float NdotV = CUM::dot(N, V);
		Float NdotL = CUM::dot(N, L);

		Float F = F0 + (1.0 - F0)*pow(1.0 - NdotH, 5.0);
		Float NDF = DistributionGGX(N, H, roughness);
		Float G = GeometrySmith(N, V, L, roughness);

		Float nominator = NDF * G * F;
		Float denominator = 4.0 * CUM::max(NdotV, 0.0) * CUM::max(NdotL, 0.0) + 0.001;
		Float specular = nominator / denominator;

		Float Ks = F;
		Float Kd = 1.0 - Ks;
		Kd *= 1.0 - metallic;

		Lo += (Kd*Albedo / PI + specular)*LightRadiance*CUM::max(NdotL, 0.0);

		return Lo;

		//Float xi1 = GetUniformRand();
		//Float xi2 = GetUniformRand();

		//Float Sqrt1MinusXi1Square = sqrt(1.0 - xi1 * xi1);
		//Float x = cos(2.0*PI*xi2)*Sqrt1MinusXi1Square;
		//Float z = sin(2.0*PI*xi2)*Sqrt1MinusXi1Square;
		//Float y = xi1;
		//CUM::Vec3f L(x, y, z);
		//Float NrDotNt = CUM::dot(N, CUM::Vec3f(0.0, 1.0, 0.0));
		//if (NrDotNt <= 1.0 - Epsilon)
		//{
		//	const CUM::Vec3f& axis = CUM::normalize(CUM::cross(CUM::Vec3f(0.0, 1.0, 0.0), N));
		//	L = CUM::RodriguesRotateCosine(axis, NrDotNt, L);
		//}
		//const CUM::Vec3f& halfDir = CUM::normalize(L + V);

		//Float NdotV = CUM::dot(N, V);
		////Float NdotH = CUM::dot(N, halfDir);
		//Float NdotH = 1.0;
		//Float roughness = 0.5;

		//Float F = F0 + (1.0 - F0)*pow(1.0 - NdotH, 5.0);
		//Float NDF = DistributionGGX(N, halfDir, roughness);
		//Float G = GeometrySmith(N, V, L, roughness);
		//Float nominator = NDF * G * F;
		//Float denominator = 4.0 * CUM::max(CUM::dot(N, V), 0.0) * CUM::max(CUM::dot(N, L), 0.0) + 0.001;
		//Float specular = nominator / denominator;

		//if (GetUniformRand() <= specular)
		//{
		//	return CUM::normalize(2.0 * N - NdotV * V);;
		//}
		//else
		//{
		//	xi1 = GetUniformRand();
		//	xi2 = GetUniformRand();

		//	Sqrt1MinusXi1Square = sqrt(1.0 - xi1 * xi1);
		//	x = cos(2.0*PI*xi2)*Sqrt1MinusXi1Square;
		//	z = sin(2.0*PI*xi2)*Sqrt1MinusXi1Square;
		//	y = xi1;
		//	CUM::Vec3f randV(x, y, z);
		//	if (NrDotNt > 1.0 - Epsilon)
		//	{
		//		return randV;
		//	}
		//	else
		//	{
		//		const CUM::Vec3f& axis = CUM::normalize(CUM::cross(CUM::Vec3f(0.0, 1.0, 0.0), N));
		//		return CUM::RodriguesRotateCosine(axis, NdotV, randV);;
		//	}
		//}
		//return CUM::normalize(2.0 * N - NdotV * V);
	}
public:
	__duel__ void testForCopyRandVec()
	{
		custd::OStream os;
		//os << "The first rand vec of the current material is:\n" << randVecs[0].x << "\t" << randVecs[0].y << "\t" << randVecs[0].z << "\t\n";
		os << "The first rand num of the current material is:\n" << randNums[0] << "\n";
	}
};

class Record
{
public:
	CUM::Point3f hitPoint;
	CUM::Normal3f normal;
	CUM::Color3f sampledColor;
	CUM::Color3f sampledLightRadiance;
	Material* sampledMaterial;
	Float times;
public:
	__duel__ Record():hitPoint(),normal(),sampledColor(),sampledLightRadiance(),times(0.0) {}
	__duel__ Record(const CUM::Point3f& _position, const CUM::Normal3f& _normal, const CUM::Color3f& _albedo, const CUM::Color3f& _lightRadiance,const Float& _time)
		:hitPoint(_position), normal(_normal), sampledColor(_albedo), sampledLightRadiance(_lightRadiance), times(_time) {}
	//Record(CUM::Point3f&& _position, CUM::Normal&& _normal, CUM::Color3f&& _albedo, Float&& _time)
	//	:position(_position), normal(_normal), albedo(_albedo), times(_time) {}
	__duel__ const Record operator=(const Record& rec)
	{
		hitPoint = rec.hitPoint;
		normal = rec.normal;
		sampledColor = rec.sampledColor;
		sampledLightRadiance = rec.sampledLightRadiance;
		sampledMaterial = rec.sampledMaterial;
		times = rec.times;
		return *this;
	}

	__duel__ ~Record()
	{

	}
};

class Ray
{
public:
	CUM::Point3f origin;
	CUM::Vec3f direction;
	Record record;
public:
	__duel__ Ray()
	{
		record.times = -1.0;
	}
public:
	__duel__ const CUM::Point3f GetEndPoint(const Float& times)
	{
		//CHECK(record.times >= 0.0, "Ray::GetEndPoint(const Float& times) error: the times can not less than 0!");
		return origin + times * direction;
	}

#ifdef RUN_ON_DEVICE
	__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
		__host__
#endif // RUN_ON_HOST
	void ProcessSampledResult()
	{
		//lightVec.Sampling(*this);
		const CUM::Vec3f L(0.0, 0.0, -1.0);
		CUM::Vec3f N(record.normal.x, record.normal.y, record.normal.z);
		CUM::Vec3f V(direction);
		CUM::Vec3f H(CUM::normalize(L + V));
		CUM::Color3f LightColor(1.0);
		CHECK(record.sampledMaterial, "Ray::InteractWithSampledResultAndShadingFromLight() error: the sampledMaterial can not be nullptr!");
		CUM::Color3f shadedLightRadience =  record.sampledMaterial->ShadeWithDirectLight(N, V, H, LightColor);
		record.sampledLightRadiance = shadedLightRadience;

		record.sampledMaterial->InteractWithRay(*this);
	}

#ifdef RUN_ON_DEVICE
	__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
	__host__
#endif // RUN_ON_HOST
	void CalculateNextRay()
	{
		origin = record.hitPoint;
		record.times = FLT_MAX;
		direction = record.sampledMaterial->GenerateNextDirection(record.normal, direction);
	}
};

struct Pixel
{
	Ushort r, g, b, a;
};

class Texture
{
public:
	Pixel* buffer;
	CUM::Vec2i size;
	Int width;
	Int height;
	Int length;
	CUM::Vec2f deltaUV;
public:
	__duel__ Texture() {}
	__duel__ Texture(const CUM::Vec2i& _size, Pixel* _buffer)
		: size(_size), width(_size.x), height(_size.y), buffer(_buffer)
	{
		length = width * height;
		deltaUV.x = 1.0 / width;
		deltaUV.y = 1.0 / height;
	}
public:
	__duel__ const CUM::Color3f GetColorRGB(const CUM::Vec2f uv) const
	{
		CHECK(uv.x <= 1.0&&uv.x >= 0.0, "Texture::GetColor(const CUM::Vec2f uv) error: the uv.x is out of range!");
		CHECK(uv.y <= 1.0&&uv.y >= 0.0, "Texture::GetColor(const CUM::Vec2f uv) error: the uv.y is out of range!");

		CHECK(size.x >= 0, "Texture::GetColor(const CUM::Vec2f uv) error: the size.x can not be zero or less than zero!");
		CHECK(size.y >= 0, "Texture::GetColor(const CUM::Vec2f uv) error: the size.y can not be zero or less than zero!");

		//CUM::Vec2f deltaEpcilon = 1.0 / size * Epsilon;
		CUM::Vec2i position;
		//position.x = floor(uv.x * width + deltaEpcilon.x);
		//position.y = floor(uv.y * height + deltaEpcilon.y);
		position.x = floor(uv.x * width);
		position.y = floor(uv.y * height);
		Int idx = position.x*height + position.y;

		CHECK(position.x <= width && position.x >= 0, "Texture::GetColor(const CUM::Vec2f uv) error: the position.x is out of range!");
		CHECK(position.y <= width && position.y >= 0, "Texture::GetColor(const CUM::Vec2f uv) error: the position.y is out of range!");
		CHECK(idx >= 0 && idx <= width * height, "Texture::GetColor(const CUM::Vec2f uv) error: the idx is out of range!");
		Float R = Float(buffer[idx].r) / 255.0;
		Float G = Float(buffer[idx].g) / 255.0;
		Float B = Float(buffer[idx].b) / 255.0;
		//Float A = Float(buffer[idx].a) / 255.0;
		return CUM::Color3f(R, G, B);
	}
public:
	__host__ Texture* copyToDevice()
	{
		Texture textureInsWithDevicePtr(*this);
		Pixel* bufferDevice;

		cudaMalloc(&bufferDevice, length * sizeof(Pixel));
		cudaMemcpy(bufferDevice, buffer, length * sizeof(Pixel), cudaMemcpyKind::cudaMemcpyHostToDevice);

		textureInsWithDevicePtr.buffer = bufferDevice;
		Texture* textureDevice = CudaInsMemCpyHostToDevice(&textureInsWithDevicePtr);
		return textureDevice;
	}

	__host__ void CopyFromDevice(Texture* device)
	{
		Texture hostTex;
		cudaError_t error =  cudaMemcpy(&hostTex, device, sizeof(Texture), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		if (error != cudaError_t::cudaSuccess)
		{
			printf("%s\n", cudaGetErrorString(error));
		}
		cudaMemcpy(buffer,hostTex.buffer, length * sizeof(Pixel), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	__duel__ void Release()
	{
		custd::OStream os;
		os << "Called Texture::Release()!\n";
		CHECK(buffer, "Texture::Release() error: buffer can not be nullptr");
		if (buffer)
		{
			delete[] buffer;
			buffer = nullptr;
		}
	}

	__host__ void Save(const char* path)
	{
		std::ofstream image(path);
		const Int maxVal = 255;
		image << "P3\n";
		image << size.x << " " << size.y << "\n";
		image << maxVal << "\n";
		for (Int i = 0; i < length; i++)
		{
			//R = round(maxVal* buffer[i].r);
			//G = round(maxVal* buffer[i].g);
			//B = round(maxVal* buffer[i].b);
			image << buffer[i].r << " " << buffer[i].g << " " << buffer[i].b << "\n";
		}
		image.close();
	}
};

class Camera
{
public:
	static Texture* RenderTargetDevice;
public:
	CUM::Point3f position;
	CUM::Vec3f direction;
	CUM::Quaternionf rotation;
	CUM::Vec2i imageSize;
	Float aspectRatio;
	Float nearPlan;
	Float farPlan;
	Int bounceTime;
	Texture* renderTarget;
public:
	__duel__ Camera() : bounceTime(64)
	{

	}

	__duel__ Camera(const Camera& cam) : position(cam.position),direction(cam.direction),rotation(cam.rotation),imageSize(cam.imageSize),aspectRatio(cam.aspectRatio),
		nearPlan(cam.nearPlan),farPlan(cam.farPlan),bounceTime(cam.bounceTime),renderTarget(cam.renderTarget)
	{

	}

	__duel__ Camera(const CUM::Point3f& _position, const CUM::Vec3f& _direction, const CUM::Quaternionf& _rotation, const CUM::Vec2i& _imageSize, const Float& _nearPlan, const Float& _farPlan, const Int& _bounceTime, Texture* _renderTarget)
		:position(_position), direction(_direction), rotation(_rotation), imageSize(_imageSize), nearPlan(_nearPlan), farPlan(_farPlan), bounceTime(_bounceTime),renderTarget(_renderTarget)
	{
		//Aspect ratio always width/height.
		aspectRatio = Float(imageSize.x) / Float(imageSize.y);
	}

	__duel__ const Camera& operator=(const Camera& cam)
	{
		bounceTime = cam.bounceTime;
		return *this;
	}

public:
	virtual Camera* copyToDevice()
	{
		Camera camWithDevicePtr(*this);
		camWithDevicePtr.renderTarget = renderTarget->copyToDevice();
		Camera* device = CudaInsMemCpyHostToDevice(&camWithDevicePtr);
		return device;
	}
	__duel__ virtual void Release()
	{
		if (renderTarget)
		{
			renderTarget->Release();
			delete renderTarget;
		}
	}
public:
	__duel__ virtual void Call()
	{
		custd::OStream os;
		os << "Called Camera"<<custd::endl;
	}
	__duel__ virtual const Ray GetRay(const CUM::Vec2f& uv)
	{
		CHECK(false, "Can not call Camera::GetRay!");
		return Ray();
	}

};

class PersCamera : public Camera
{
public:
	Float fovH;
public:

	__duel__ PersCamera() {}

	__duel__ PersCamera(const CUM::Point3f& _position, const CUM::Vec3f& _direction, const CUM::Quaternionf& _rotation, const CUM::Vec2i& _imageSize, const Float& _nearPlan, const Float& _farPlan, const Int& _bounceTime, const Float& _fovH, Texture* _renderTarget)
		: Camera(_position, _direction, _rotation, _imageSize, _nearPlan, _farPlan, _bounceTime,_renderTarget), fovH(_fovH) {}
public:
	__host__ virtual PersCamera* copyToDevice() override
	{
		PersCamera persCamWithDevicePtr(*this);
		persCamWithDevicePtr.renderTarget = renderTarget->copyToDevice();
		RenderTargetDevice = persCamWithDevicePtr.renderTarget;
		PersCamera* device = CudaInsMemCpyHostToDevice(&persCamWithDevicePtr);
		return device;
	}
	__duel__ virtual void Release() override
	{
		custd::OStream os;
		os << "Called PersCamera::Release()!\n";
		CHECK(renderTarget, "PersCamera::Release() error: RenderTarget can not be nullptr!");
		if (renderTarget)
		{
			renderTarget->Release();
			delete renderTarget;
			renderTarget = nullptr;
		}

	}
public:
	__duel__ virtual void Call() override
	{
		custd::OStream os;
		os << "Called PersCamera!\n";
	}
	__duel__ virtual const Ray GetRay(const CUM::Vec2f& uv) override
	{
		CHECK(aspectRatio != 0.0, "PersCamera::GetRay(const CUM::Vec2f& uv) error: the aspectRatio can not be 0!");
		Float unitWidth = 2.0 * tan(0.5 * fovH);
		Float unitHeight = unitWidth / aspectRatio;
		CUM::Vec2f flippedUV(uv.x, 1.0 - uv.y);
		CUM::Vec2f unitPosFactor = (flippedUV - 0.5);
		Float uintX = unitPosFactor.x * unitWidth;
		Float uintY = unitPosFactor.y * unitHeight;

		CUM::Point3f directionPos(uintX, uintY, 1.0);
		Ray result;
		result.origin = position;
		//result.direction = CUM::applyQuaTransform(rotation, CUM::normalize(directionPos - position));
		result.direction = CUM::normalize(directionPos - position);

		return result;
	}
};

#endif // !__TOOLS__CUH__
