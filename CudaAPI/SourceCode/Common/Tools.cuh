#ifndef __RAY__CUH__
#define __RAY__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "Cuda3DMath.cuh"

class Record
{
public:
	CUM::Point3f position;
	CUM::Normal3f normal;
	CUM::Color3f albedo;
	Float times;
public:
	__duel__ Record():position(CUM::Point3f()),normal(),albedo(CUM::Color3f()),times(0.0) {}
	__duel__ Record(const CUM::Point3f& _position, const CUM::Normal3f& _normal, const CUM::Color3f& _albedo, const Float& _time)
		:position(_position), normal(_normal), albedo(_albedo), times(_time) {}
	//Record(CUM::Point3f&& _position, CUM::Normal&& _normal, CUM::Color3f&& _albedo, Float&& _time)
	//	:position(_position), normal(_normal), albedo(_albedo), times(_time) {}
	__duel__ const Record operator=(const Record& rec)
	{
		position = rec.position;
		normal = rec.normal;
		albedo = rec.albedo;
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
public:
	__duel__ Camera(const CUM::Point3f& _position, const CUM::Vec3f& _direction, const CUM::Quaternionf& _rotation, const CUM::Vec2i& _imageSize, const Float& _nearPlan, const Float& _farPlan)
		:position(_position), direction(_direction), rotation(_rotation), imageSize(_imageSize), nearPlan(_nearPlan), farPlan(_farPlan)
	{
		//Aspect ratio always width/height.
		aspectRatio = Float(imageSize.x) / Float(imageSize.y);
	}
};

class PersCamera : public Camera
{
public:
	Float fovH;
public:
	__duel__ PersCamera(const CUM::Point3f& _position, const CUM::Vec3f& _direction, const CUM::Quaternionf& _rotation, const CUM::Vec2i& _imageSize, const Float& _nearPlan, const Float& _farPlan, const Float& _fovH)
		: Camera(_position, _direction, _rotation, _imageSize, _nearPlan, _farPlan), fovH(_fovH) {}

};

#endif // !__RAY__CUH__
