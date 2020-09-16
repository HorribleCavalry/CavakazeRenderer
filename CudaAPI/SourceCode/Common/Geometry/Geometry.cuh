#ifndef __GEOMETRY__CUH__
#define __GEOMETRY__CUH__

#include "../Cuda3DMath.cuh"
#include "../Ray.cuh"
#include "../../CudaSTD/cuiostream.cuh"

#define Epsilon 0.0078125

class Geometry
{
public:
	CUM::Point3f centroid;
protected:
	Float area;
	Float volume;
public:
	__duel__ Geometry() : centroid() {}
	__duel__ Geometry(const CUM::Point3f& _centroid) : centroid(_centroid) {}
public:
	__duel__ virtual const Bool HitTest(Ray& inputRay)
	{
		CHECK(false, "Use Geometry::HitTest is not permitted!");
		return false;
	}
	__duel__ virtual const Float GetArea()
	{
		CHECK(false, "Use Geometry::GetArea is not permitted!");
		return 0.0;
	}
	__duel__ virtual const Float GetVolume()
	{
		CHECK(false, "Use Geometry::GetVolume is not permitted!");
		return 0.0;
	}
};

class Sphere : public Geometry
{
public:
	Float radius;

public:
	__duel__ virtual const Bool HitTest(Ray& inputRay) override
	{
		CUM::Vec3f origin(inputRay.origin.x, inputRay.origin.y, inputRay.origin.z);
		Float a = CUM::dot(inputRay.direction, inputRay.direction);
		Float b = 2.0 * dot(inputRay.direction, origin);
		Float c = CUM::dot(origin, origin) - radius * radius;
		Float discriminant = b * b - 4.0*a*c;
		Float times = 0.0;
		if (discriminant > 0.0)
		{
			times = 0.5 *(-b - discriminant) / a;
			if (times<FLT_MAX && times>Epsilon)
			{
				inputRay.record.times = times;
				CUM::Point3f endPoint = inputRay.GetEndPoint(times);

			}
		}

		return false;
	}
	__duel__ virtual const Float GetArea() override
	{
		const custd::OStream os; os << "Called Sphere GetArea!" << custd::endl;
		return 0.0;
	}
	__duel__ virtual const Float GetVolume() override
	{
		return 0.0;
	}
};

class Box : public Geometry
{
public:
	CUM::Point3f leftBottom;
	CUM::Quaternionf rotation;
private:
	CUM::Vec3f geoInfo;
public:

	__duel__ Box() : leftBottom(-1.0, 0.0, 5.0), rotation(0, 0, 0, 1), Geometry(CUM::Point3f(0.0, 1.0, 6)) {}


	__duel__ Box(CUM::Point3f& _centroid, const CUM::Point3f&_leftBottom, const CUM::Quaternionf& _rotation)
		:leftBottom(_leftBottom), rotation(_rotation), Geometry(_centroid)
	{
		geoInfo = 2.0 * CUM::abs(centroid - leftBottom);
		area = 2.0 * (geoInfo.x*geoInfo.y + geoInfo.x*geoInfo.z + geoInfo.y*geoInfo.z);
		volume = geoInfo.x * geoInfo.y * geoInfo.z;
	}
public:
	__duel__ virtual const Bool HitTest(Ray& inputRay) override
	{
		return false;
	}
	__duel__ virtual const Float GetArea() override
	{
		const custd::OStream os; os << "Called Box GetArea!" << custd::endl;
		return area;
	}
	__duel__ virtual const Float GetVolume() override
	{
		return volume;
	}
};

#endif // !__GEOMETRY__CUH__
