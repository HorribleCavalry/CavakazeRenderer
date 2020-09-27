#ifndef __GEOMETRY__CUH__
#define __GEOMETRY__CUH__

#include "../Cuda3DMath.cuh"
#include "../Tools.cuh"
#include "../../CudaSTD/cuiostream.cuh"
#include "../CudaPrimitivesVector.cuh"
#include "../../CudaSTD/cuvector.cuh"

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
	__duel__ virtual const Bool HitTest(Ray& ray)
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
public:
	__host__ virtual Geometry* copyToDevice()
	{
		return CudaInsMemCpyHostToDevice(this);
	}
	__duel__ virtual void Release()
	{
		custd::OStream os;
		os << "Called Geometry::Release()!\n";
	}
public:
	__duel__ virtual void Call()
	{
		custd::OStream os;
		os << "Called Geometry::Call();\n";
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
public:
	virtual Sphere* copyToDevice() override
	{
		return CudaInsMemCpyHostToDevice(this);
	}
	__duel__ virtual void Release() override
	{
		custd::OStream os;
		os << "Called Sphere::Release()!\n";
	}
public:
	__duel__ virtual void Call() override
	{
		custd::OStream os;
		os << "Called Sphere::Call();\n";
	}
};

//Orientd Box
class BBox : public Geometry
{
private:
	CUM::Vec3f extent;
	CUM::Point3f pMin;
	CUM::Point3f pMax;
public:

	__duel__ BBox() : Geometry(CUM::Point3f(0.0, 1.0, 6)) {}


	__duel__ BBox(CUM::Point3f& _centroid, const CUM::Vec3f& _extent)
		: extent(_extent), Geometry(_centroid)
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
		CHECK(!v.IsZero(), "BBox::GetNormal error: the v can not be a zero vec!");
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
	__duel__ virtual const Bool HitTest(Ray& rayB) override
	{
		CUM::Vec3f directionB = rayB.direction;
		CUM::Point3f originB = rayB.origin - centroid;

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
			rayB.record.times = t0;
			CUM::Point3f hitPositionB(originB + t0 * directionB);
			CUM::Normal3f normalB(GetNormal(hitPositionB - 0.0));
			rayB.record.normal = CUM::Vec3f(normalB.x, normalB.y, normalB.z);
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
public:
	virtual BBox* copyToDevice() override
	{
		return CudaInsMemCpyHostToDevice(this);
	}
	__duel__ virtual void Release() override
	{
		custd::OStream os;
		os << "Called BBox::Release()!\n";
	}
public:
	__duel__ virtual void Call() override
	{
		custd::OStream os;
		os << "Called BBox::Call();\n";
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
public:
	virtual OBox* copyToDevice() override
	{
		return CudaInsMemCpyHostToDevice(this);
	}
	__duel__ virtual void Release() override
	{
		custd::OStream os;
		os << "Called OBox::Release()!\n";
	}
public:
	__duel__ virtual void Call() override
	{
		custd::OStream os;
		os << "Called OBox::Call();\n";
	}
};

class Triangle : public Geometry
{
public:
	CUM::Point3f points[3];
	CUM::Normal3f normal;
public:
	__duel__ Triangle()
	{

	}

	__duel__ Triangle(const CUM::Point3f _points[3])
	{
		for (Int i = 0; i < 3; i++)
		{
			points[i] = _points[i];
		}
		normal = CUM::normalize(CUM::cross(points[1] - points[0], points[2] - points[0]));
	}
public:
	__duel__ virtual const Bool HitTest(Ray& ray) override
	{
		Float a = points[0].x - points[1].x, b = points[0].x - points[2].x, c = ray.direction.x, d = points[0].x - ray.origin.x;
		Float e = points[0].y - points[1].y, f = points[0].y - points[2].y, g = ray.direction.y, h = points[0].y - ray.origin.y;
		Float i = points[0].z - points[1].z, j = points[0].z - points[2].z, k = ray.direction.z, l = points[0].z - ray.origin.z;

		Float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		Float q = g * i - e * k, s = e * j - f * i;

		double inv_denom = 1.0 / (a * m + b * q + c * s);

		double e1 = d * m - b * n - c * p;
		double beta = e1 * inv_denom;

		if (beta < 0.0)
			return (false);

		double r = r = e * l - h * i;
		double e2 = a * n + d * q + c * r;
		double gamma = e2 * inv_denom;

		if (gamma < 0.0)
			return false;

		if (beta + gamma > 1.0)
			return false;

		double e3 = a * p - b * r + d * s;
		double t = e3 * inv_denom;

		if (t < Epsilon)
			return false;

		ray.record.times = t;
		ray.record.normal = normal;
		ray.record.position = ray.GetEndPoint(t);
		return true;
	}
	__duel__ virtual const Float GetArea()
	{
		return 0.5 * CUM::norm(CUM::cross(points[1] - points[0], points[2] - points[0]));
	}
	__duel__ virtual const Float GetVolume()
	{
		CHECK(false, "The triangle do not have volume!");
		return 0.0;
	}
public:
	__host__ virtual Triangle* copyToDevice() override
	{
		return CudaInsMemCpyHostToDevice(this);
	}
	__duel__ virtual void Release() override
	{
		custd::OStream os;
		os << "Called Triangle::Release()!\n";
	}
public:
	__duel__ virtual void Call() override
	{
		custd::OStream os;
		os << "Called Triangle::Call();\n";
	}
};

class Mesh
{
public:
	CUM::PrimitiveVector<Geometry> primitives;
};

class Object
{
public:
	CUM::Transform transform;
	Mesh mesh;
	BBox bBox;
	Material material;
};

struct HierarchyTreeNode
{
	Object object;
	custd::cuvector<HierarchyTreeNode*> childNodes;
};

class HierarchyTree
{
private:
	static HierarchyTreeNode* headNod;
public:
	__duel__ const HierarchyTreeNode* GetInstance()
	{
		if (!headNod)
			headNod = new HierarchyTreeNode;
		return headNod;
	}
};


class Scene
{
public:
	PersCamera*  camera;

	//HierarchyTree hierarchyTree;
	CUM::PrimitiveVector<Geometry>* primitivesVectorPtr;
public:
	__duel__ void Call()
	{
		custd::OStream os;
		os << "Called Scene::Call();\n";
	}


public:

	__duel__ Scene()
	{

	}

	__duel__ Scene(PersCamera* _camera, CUM::PrimitiveVector<Geometry>* _primitivesVectorPtr)
		:camera(_camera), primitivesVectorPtr(_primitivesVectorPtr)
	{
		
	}
public:
	__duel__ void Rendering()
	{
		//CUM::Vec2i size(camera->imageSize);
		//Int totalNum = size.x*size.y;

		//Int threadNum = 1024;
		//Int blockNum = Int(floor((Float(totalNum)/threadNum)) + Epsilon);
		//rendering <<<blockNum, threadNum >>>
		//	(cameraDevice,
		//	primitivesVector_device);
	}

	__duel__ void EndRendering()
	{

	}

public:
	Scene* copyToDevice()
	{
		Scene sceneInsWithDevicePtr(*this);

		PersCamera* cameraDevice = camera->copyToDevice();
		CUM::PrimitiveVector<Geometry>* primitivesVectorPtrDevice = primitivesVectorPtr->copyToDevice();

		sceneInsWithDevicePtr.camera = cameraDevice;
		sceneInsWithDevicePtr.primitivesVectorPtr = primitivesVectorPtrDevice;

		Scene* sceneDevice = CudaInsMemCpyHostToDevice(&sceneInsWithDevicePtr);
		return sceneDevice;
	}
	__duel__ void Release()
	{
		custd::OStream os;
		os << "Called Scene::Release()!\n";
		CHECK(camera, "Scene::Release() error: camera can not be nullptr!");
		CHECK(primitivesVectorPtr, "Scene::Release() error: primitivesVectorPtr can not be nullptr!");
		if (camera)
		{
			camera->Release();
			camera = nullptr;
		}
		if (primitivesVectorPtr)
		{
			primitivesVectorPtr->Release();
			primitivesVectorPtr = nullptr;
		}
		
	}
public:
	void AddPrimitive(Geometry& geo)
	{
		(*primitivesVectorPtr).push_back(geo);
	}
public:


};

__global__ void rendering(Scene* scene)
{
	Int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;
	PersCamera& camera = *scene->camera;
	CUM::PrimitiveVector<Geometry>& primitiveVec = *(scene->primitivesVectorPtr);
	CUM::Vec2i size = camera.renderTarget->size;

	Int x = globalIdx % size.x;
	Int y = globalIdx / size.x;

	Float u = Float(x) / Float(size.x);
	Float v = Float(y) / Float(size.y);
	CUM::Vec2f uv(u,v);
	
	//camera.Call();
	Ray ray = camera.GetRay(uv);

	//CUM::Color3f resultColor(1.0);
	//CUM::Color3f tempColor(1.0);

	//Bool haveHitPrimitives = false;

	//for (Int i = 0; i < camera.sampleTime; i++)
	//{

	//	if (primitiveVec.HitTest(ray))
	//	{
	//		tempColor = ray.record.sampledColor;
	//		resultColor *= tempColor;
	//		ray = ray.CalculateNextRay();
	//	}
	//	else
	//	{
	//		resultColor *= CUM::Color3f(1.0, 0.0, 1.0);
	//		break;
	//	}
	//}
	//camera.renderTarget->buffer[globalIdx].r = round(resultColor.r);
	//camera.renderTarget->buffer[globalIdx].g = round(resultColor.g);
	//camera.renderTarget->buffer[globalIdx].b = round(resultColor.b);

	camera.renderTarget->buffer[globalIdx].r = Ushort(round(255.0 * uv.x));
	camera.renderTarget->buffer[globalIdx].g = Ushort(round(255.0 * uv.y));
	camera.renderTarget->buffer[globalIdx].b = 0;
}


#endif // !__GEOMETRY__CUH__
