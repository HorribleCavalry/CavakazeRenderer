#ifndef __RAY__CUH__
#define __RAY__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "Cuda3DMath.cuh"

class Material
{

};

class Record
{
public:
	CUM::Point3f position;
	CUM::Normal3f normal;
	CUM::Color3f sampledColor;
	Material sampledMaterial;
	Float times;
public:
	__duel__ Record():position(CUM::Point3f()),normal(),sampledColor(CUM::Color3f()),times(0.0) {}
	__duel__ Record(const CUM::Point3f& _position, const CUM::Normal3f& _normal, const CUM::Color3f& _albedo, const Float& _time)
		:position(_position), normal(_normal), sampledColor(_albedo), times(_time) {}
	//Record(CUM::Point3f&& _position, CUM::Normal&& _normal, CUM::Color3f&& _albedo, Float&& _time)
	//	:position(_position), normal(_normal), albedo(_albedo), times(_time) {}
	__duel__ const Record operator=(const Record& rec)
	{
		position = rec.position;
		normal = rec.normal;
		sampledColor = rec.sampledColor;
		times = rec.times;
		return *this;
	}
	//const Record operator=(Record&& rec)
	//{
	//	position = rec.position;
	//	normal = rec.normal;
	//	albedo = rec.albedo;
	//	times = rec.times;
	//}
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
	__duel__ const CUM::Point3f GetEndPoint(const Float& times)
	{
		return origin + times * direction;
	}
	//To do...
	__duel__ const Ray CalculateNextRay()
	{
		return Ray();
	}
};

class Texture
{
public:
	CUM::Color3f* buffer;
	CUM::Vec2i size;
	Int width;
	Int height;
	Int length;
public:
	__duel__ Texture() {}
	__duel__ Texture(const CUM::Vec2i& _size)
		: size(_size), width(_size.x), height(_size.y)
	{
		length = width * height;
		buffer = new CUM::Color3f[width * height];
	}
};

class Camera
{
public:
	CUM::Point3f position;
	CUM::Vec3f direction;
	CUM::Quaternionf rotation;
	CUM::Vec2i imageSize;
	Float aspectRatio;
	Float nearPlan;
	Float farPlan;
	Int sampleTime;
	Texture renderTarget;
public:
	__duel__ Camera()
	{

	}

	__duel__ Camera(const Camera& cam) : position(cam.position),direction(cam.direction),rotation(cam.rotation),imageSize(cam.imageSize),aspectRatio(cam.aspectRatio),
		nearPlan(cam.nearPlan),farPlan(cam.farPlan),sampleTime(cam.sampleTime),renderTarget(cam.renderTarget)
	{

	}

	__duel__ Camera(const CUM::Point3f& _position, const CUM::Vec3f& _direction, const CUM::Quaternionf& _rotation, const CUM::Vec2i& _imageSize, const Float& _nearPlan, const Float& _farPlan, const Int& _sampleTime, Texture _renderTarget)
		:position(_position), direction(_direction), rotation(_rotation), imageSize(_imageSize), nearPlan(_nearPlan), farPlan(_farPlan), sampleTime(_sampleTime),renderTarget(_renderTarget)
	{
		//Aspect ratio always width/height.
		aspectRatio = Float(imageSize.x) / Float(imageSize.y);
	}

	__duel__ const Camera& operator=(const Camera& cam)
	{
		sampleTime = cam.sampleTime;
	}

public:
	__duel__ const Ray GetRay(const CUM::Vec2f& uv)
	{
		return Ray();
	}

	virtual Camera* copyToDevice()
	{
		Camera result(*this);
		Int bufferLength = renderTarget.length;
		cudaMalloc(&result.renderTarget.buffer, bufferLength * sizeof(CUM::Color3f));
		return nullptr;
	}
public:
	__duel__ virtual void Call()
	{
		custd::OStream os;
		os << "Called Camera"<<custd::endl;
	}
};

class PersCamera : public Camera
{
public:
	Float fovH;
public:

	__duel__ PersCamera() {}

	__duel__ PersCamera(const CUM::Point3f& _position, const CUM::Vec3f& _direction, const CUM::Quaternionf& _rotation, const CUM::Vec2i& _imageSize, const Float& _nearPlan, const Float& _farPlan, const Int& _sampleTime, const Float& _fovH, Texture _renderTarget)
		: Camera(_position, _direction, _rotation, _imageSize, _nearPlan, _farPlan, _sampleTime,_renderTarget), fovH(_fovH) {}
	virtual PersCamera* copyToDevice() override
	{
		PersCamera persCamWithDevicePtr(*this);
		PersCamera* device = CudaInsMemCpyHostToDevice(&persCamWithDevicePtr);
		return device;
	}
public:
	__duel__ virtual void Call() override
	{
		custd::OStream os;
		os << "Called PersCamera" << custd::endl;
	}
};

#endif // !__RAY__CUH__
