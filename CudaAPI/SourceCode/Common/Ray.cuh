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
	__duel__ Record():position(CUM::Point3f()),normal(CUM::Normal()),albedo(CUM::Color3f()),times(0.0) {}
	__duel__ Record(const CUM::Point3f& _position, const CUM::Normal& _normal, const CUM::Color3f& _albedo, const Float& _time)
		:position(_position), normal(_normal), albedo(_albedo), times(_time) {}
	//Record(CUM::Point3f&& _position, CUM::Normal&& _normal, CUM::Color3f&& _albedo, Float&& _time)
	//	:position(_position), normal(_normal), albedo(_albedo), times(_time) {}
	__duel__ const Record operator=(const Record& rec)
	{
		position = rec.position;
		normal = rec.normal;
		albedo = rec.albedo;
		times = rec.times;
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
	const CUM::Point3f GetEndPoint(const Float& times)
	{
		return origin + times * direction;
	}
};

#endif // !__RAY__CUH__
