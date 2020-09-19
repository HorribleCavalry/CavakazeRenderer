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

		if (discriminant < 0.0)
		{
			return false;
		}
		else
		{
			times = 0.5 *(-b - discriminant) / a;
			if (times < Epsilon)
			{
				times = 0.5 *(-b + discriminant) / a;
				if (times < Epsilon)
					return false;
			}
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

//Orientd Box
class OBox : public Geometry
{
public:
	CUM::Quaternionf rotation;
private:
	CUM::Vec3f extent;
	CUM::Point3f pMin;
	CUM::Point3f pMax;
public:

	__duel__ OBox() : rotation(0, 0, 0, 1), Geometry(CUM::Point3f(0.0, 1.0, 6)) {}


	__duel__ OBox(CUM::Point3f& _centroid, const CUM::Vec3f& _extent, const CUM::Quaternionf& _rotation)
		: extent(_extent), rotation(_rotation), Geometry(_centroid)
	{
		pMin = -extent;
		pMax = extent;
		CUM::Vec3f pMax;
		area = 2.0 * (extent.x*extent.y + extent.x*extent.z + extent.y*extent.z);
		volume = extent.x * extent.y * extent.z;
	}
private:
	const Int GetUnitVal(const Float& val)
	{
		return val >= 0.0 ? 1.0 : -1.0;
	}
	const CUM::Normal3f GetNormal(const CUM::Vec3f& v)
	{
		CHECK(!v.IsZero(), "OBox::GetNormal error: the v can not be a zero vec!");
		Float unit;
		switch (v.MaxAbsIdx())
		{
		case 0:
		{
			unit = GetUnitVal(v.x);
			return CUM::Normal3f(unit, 0.0, 0.0);
		}break;

		case 1:
		{
			unit = GetUnitVal(v.y);
			return CUM::Normal3f(0.0, unit, 0.0);
		}break;

		case 2:
		{
			unit = GetUnitVal(v.z);
			return CUM::Normal3f(0.0, 0.0, unit);
		}break;

		default:
			CHECK(false, "OBox::GetNormal error: can not run switch::default!");
			break;
		}
		CHECK(false, "OBox::GetNormal error: can not run switch::default!");
		return CUM::Normal3f(1.0, 0.0, 0.0);
	}
public:
	__duel__ virtual const Bool HitTest(Ray& ray) override
	{
		CUM::Vec3f directionB = CUM::applyInvQuaTransform(rotation,ray.direction);
		CUM::Point3f originB = CUM::applyInvQuaTransform(rotation, ray.origin - centroid);

		Float ox = originB.x; Float oy = originB.y; Float oz = originB.z;
		Float dx = directionB.x; Float dy = directionB.y; Float dz = directionB.z;

		Float tx_min, ty_min, tz_min;
		Float tx_max, ty_max, tz_max;

		Float a = 1.0 / dx;
		if (a >= 0)
		{
			tx_min = (pMin.x - ox) * a;
			tx_max = (pMax.x - ox) * a;
		}
		else
		{
			tx_min = (pMax.x - ox) * a;
			tx_max = (pMin.x - ox) * a;
		}

		Float b = 1.0 / dy;
		if (b >= 0)
		{
			ty_min = (pMin.y - oy) * b;
			ty_max = (pMax.y - oy) * b;
		}
		else
		{
			ty_min = (pMax.y - oy) * b;
			ty_max = (pMin.y - oy) * b;
		}

		Float c = 1.0 / dy;
		if (b >= 0)
		{
			tz_min = (pMin.z - oz) * c;
			tz_max = (pMax.z - oz) * c;
		}
		else
		{
			tz_min = (pMax.z - oz) * c;
			tz_max = (pMin.z - oz) * c;
		}

		Float t0, t1;

		t0 = tx_min > ty_min ? tx_min : ty_min;
		t0 = tz_min > 0.0 ? tz_min : t0;

		t1 = tx_max < ty_max ? tx_max : ty_max;
		t1 = tz_max < t1 ? tz_max : t1;

		Bool isHit = t0 < t1 && t1 > Epsilon;

		if (isHit)
		{
			ray.record.times = t0;
			CUM::Point3f hitPositionB(originB + t0 * directionB);
			CUM::Normal3f normalB(GetNormal(hitPositionB - 0.0));
			CUM::Vec3f normal(CUM::applyQuaTransform(rotation, CUM::Vec3f(normalB.x, normalB.y, normalB.z)));
			ray.record.normal = normal;
		}

		return isHit;
	}
	__duel__ virtual const Float GetArea() override
	{
		return area;
	}
	__duel__ virtual const Float GetVolume() override
	{
		return volume;
	}
};

#endif // !__GEOMETRY__CUH__
