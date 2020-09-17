#ifndef __GEOMETRY__CUH__
#define __GEOMETRY__CUH__

#include "../Cuda3DMath.cuh"
#include "../Tools.cuh"
#include "../../CudaSTD/cuiostream.cuh"

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
	__duel__ Sphere() : Geometry(), radius(1.0) {}
	__duel__ Sphere(const CUM::Point3f& _centroid, const Float& _radius) : Geometry(), radius(_radius)
	{
		area = 4.0 * PI * radius * radius;
		volume = 4.0 * PI * radius * radius * radius / 3.0;
	}
public:
	__duel__ virtual const Bool HitTest(Ray& ray) override
	{
		CHECK(radius > 0.0, "Sphere::HitTest error: radius can not be 0!");

		CUM::Vec3f origin(ray.origin.x, ray.origin.y, ray.origin.z);
		Float a = CUM::dot(ray.direction, ray.direction);
		Float b = 2.0 * dot(ray.direction, origin);
		Float c = CUM::dot(origin, origin) - radius * radius;
		Float discriminant = b * b - 4.0*a*c;
		Float times = 0.0;

		CUM::Point3f endPoint;
		CUM::Normal3f normal;

		if (discriminant > 0.0)
		{
			times = 0.5 *(-b - discriminant) / a;
			if (times < Epsilon)
				times = 0.5 *(-b + discriminant) / a;
			
			ray.record.times = times;
			endPoint = ray.GetEndPoint(times);
			normal = (endPoint - centroid) / radius;
			ray.record.times = times;
			ray.record.position = endPoint;
			ray.record.normal = normal;
			return true;
		}

		return false;
	}
	__duel__ virtual const Float GetArea() override
	{
		return area;
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
	CUM::Vec3f extent;
public:

	__duel__ Box() : leftBottom(-1.0, 0.0, 5.0), rotation(0, 0, 0, 1), Geometry(CUM::Point3f(0.0, 1.0, 6)) {}


	__duel__ Box(CUM::Point3f& _centroid, const CUM::Point3f&_leftBottom, const CUM::Quaternionf& _rotation)
		:leftBottom(_leftBottom), rotation(_rotation), Geometry(_centroid)
	{
		extent = 2.0 * CUM::abs(centroid - leftBottom);
		area = 2.0 * (extent.x*extent.y + extent.x*extent.z + extent.y*extent.z);
		volume = extent.x * extent.y * extent.z;
	}
public:
	__duel__ virtual const Bool HitTest(Ray& ray) override
	{
		CUM::Vec3f directionB = CUM::applyQuaTransform(CUM::conjugate(rotation),ray.direction);

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
