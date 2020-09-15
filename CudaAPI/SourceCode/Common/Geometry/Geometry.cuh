#ifndef __GEOMETRY__CUH__
#define __GEOMETRY__CUH__

#include "../Cuda3DMath.cuh"
#include "../Ray.cuh"

class Geometry
{
public:
	CUM::Point3f centroid;
	Float area;
	Float volume;
public:
	virtual const Record HitTest(Ray& inputRay)
	{
		CHECK(false, "Use Geometry::HitTest is not permitted!");
		return Record();
	}
	virtual const Float GetArea()
	{
		CHECK(false, "Use Geometry::GetArea is not permitted!");
		return 0.0;
	}
	virtual const Float GetVolume()
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
	virtual const Record HitTest(Ray& inputRay) override
	{
		return inputRay.record;
	}
	virtual const Float GetArea() override
	{
		return 0.0;
	}
	virtual const Float GetVolume() override
	{
		return 0.0;
	}
};

#endif // !__GEOMETRY__CUH__
