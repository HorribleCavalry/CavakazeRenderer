﻿#ifndef __GEOMETRY__CUH__
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
	__duel__ Sphere(const CUM::Point3f& _centroid, const Float& _radius) : Geometry(_centroid), radius(_radius)
	{
		area = 4.0 * PI * radius * radius;
		volume = 4.0 * PI * radius * radius * radius / 3.0;
	}
public:
	__duel__ virtual const Bool HitTest(Ray& ray) override
	{
		CHECK(radius > 0.0, "Sphere::HitTest error: radius can not less than 0!");

		CUM::Vec3f CO(ray.origin - centroid);
		Float A = CUM::dot(ray.direction, ray.direction);
		Float B = 2.0 * dot(ray.direction, CO);
		Float C = CUM::dot(CO, CO) - radius * radius;
		Float discriminant = B * B - 4.0*A*C;
		Float sqrtDisc;
		Float times = 0.0;

		CUM::Point3f endPoint;
		CUM::Normal3f normal;

		if (discriminant < 0.0)
		{
			return false;
		}
		else
		{
			sqrtDisc = sqrt(discriminant);
			times = 0.5 *(-B - sqrtDisc) / A;
			if (times < Epsilon)
			{
				times = 0.5 *(-B + sqrtDisc) / A;
				if (times < Epsilon)
					return false;
			}
			ray.record.times = times;
			endPoint = ray.GetEndPoint(times);
			normal = CUM::normalize(endPoint - centroid);

			ray.record.sampledColor = CUM::Color3f(0.8, 0.8, 0.8);
			ray.record.times = times;
			ray.record.position = endPoint;
			ray.record.normal = normal;
			return true;
		}
		CHECK(false, "Sphere::HitTest(Ray& ray) error: it can not be here!");
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

//Bounding Box
class BBox : public Geometry
{
private:
	CUM::Vec3f extent;
	CUM::Point3f pMin;
	CUM::Point3f pMax;
public:

	__duel__ BBox() : Geometry(CUM::Point3f(0.0)) {}

	__duel__ BBox(const CUM::Point3f& _pMin, CUM::Point3f& _pMax)
		:pMin(_pMin), pMax(_pMax)
	{
		centroid = 0.5 * (pMin + pMax);
		extent = 0.5 * (pMax - pMin);
	}


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
	__duel__ const Int GetUnitVal(const Float& val)
	{
		CHECK(val != 0.0, "BBox::GetUnitVal(const Float& val) error: the val can not be zero!");
		return val > 0.0 ? 1.0 : -1.0;
	}
	__duel__ const CUM::Normal3f GetNormal(const CUM::Vec3f& v)
	{
		CHECK(!v.IsZero(), "BBox::GetNormal error: the v can not be a zero vec!");
		CUM::Vec3f normalizedV(CUM::normalize(v));
		Float unit;
		switch (normalizedV.MaxAbsIdx())
		{
		case 0:
		{
			unit = GetUnitVal(normalizedV.x);
			return CUM::Normal3f(unit, 0.0, 0.0);
		}break;

		case 1:
		{
			unit = GetUnitVal(normalizedV.y);
			return CUM::Normal3f(0.0, unit, 0.0);
		}break;

		case 2:
		{
			unit = GetUnitVal(normalizedV.z);
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
		Float xn, yn, zn;

		Float a = 1.0 / dx;
		if (a >= 0)
		{
			tx_min = (pMin.x - ox) * a;
			tx_max = (pMax.x - ox) * a;
			xn = 1.0;
		}
		else
		{
			tx_min = (pMax.x - ox) * a;
			tx_max = (pMin.x - ox) * a;
			xn = -1.0;
		}

		Float b = 1.0 / dy;
		if (b >= 0)
		{
			ty_min = (pMin.y - oy) * b;
			ty_max = (pMax.y - oy) * b;
			yn = 1.0;
		}
		else
		{
			ty_min = (pMax.y - oy) * b;
			ty_max = (pMin.y - oy) * b;
			yn = -1.0;
		}

		Float c = 1.0 / dz;
		if (c >= 0)
		{
			tz_min = (pMin.z - oz) * c;
			tz_max = (pMax.z - oz) * c;
			zn = 1.0;
		}
		else
		{
			tz_min = (pMax.z - oz) * c;
			tz_max = (pMin.z - oz) * c;
			zn = -1.0;
		}

		Float t0, t1;

		CUM::Normal3f resultNormal;

		if (tx_min > ty_min)
		{
			t0 = tx_min;
			resultNormal = xn * CUM::Vec3f(-1.0, 0.0, 0.0);
		}
		else
		{
			t0 = ty_min;
			resultNormal = yn * CUM::Vec3f(0.0, -1.0, 0.0);
		}

		if (tz_min > t0)
		{
			resultNormal = zn * CUM::Vec3f(0.0, 0.0, -1.0);
		}
		t0 = tz_min > t0 ? tz_min : t0;

		//t0 = tx_min > ty_min ? tx_min : ty_min;
		//t0 = tz_min > Epsilon ? tz_min : t0;

		t1 = tx_max < ty_max ? tx_max : ty_max;
		t1 = tz_max < t1 ? tz_max : t1;


		

		Bool isHit = t0 < t1 && t1 > Epsilon;

		if (isHit)
		{
			rayB.record.times = t0;
			rayB.record.position = rayB.GetEndPoint(t0);
			rayB.record.sampledColor = CUM::Color3f(0.8, 0.8, 0.8);
			rayB.record.normal = resultNormal;
		}
		else
		{
			rayB.record.times = FLT_MAX;
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
	__duel__ const Int GetUnitVal(const Float& val)
	{
		return val >= 0.0 ? 1.0 : -1.0;
	}
	__duel__ const CUM::Normal3f GetNormal(const CUM::Vec3f& v)
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
	CUM::PrimitiveVector<Geometry>* primitivesVec;
	Material* material;
private:
	BBox BboX;
public:
	__duel__ Mesh(CUM::PrimitiveVector<Geometry>* _primitivesVec, Material* _material)
		:primitivesVec(_primitivesVec), material(_material)
	{
		CUM::Point3f pMin, pMax;
		primitivesVec->GetMinMax(pMin, pMax);
		BboX = BBox(pMin, pMax);
	}

	__duel__ const Bool HitTest(Ray& ray)
	{
		ray.record.sampledMaterial = material;
		return primitivesVec->HitTest(ray);
	}
public:
	__host__ Mesh* copyToDevice()
	{
		Mesh meshInsWithDevicePtr(*this);

		CUM::PrimitiveVector<Geometry>* primitivesVecDevice = primitivesVec->copyToDevice();
		Material* materialDevice = material->copyToDevice();

		meshInsWithDevicePtr.primitivesVec = primitivesVecDevice;
		meshInsWithDevicePtr.material = materialDevice;

		Mesh* meshDevice = CudaInsMemCpyHostToDevice(&meshInsWithDevicePtr);
		return meshDevice;
	}
	__duel__ void Release()
	{
		custd::OStream os;
		os << "Called Mesh::Release()!\n";
		primitivesVec->Release();
		delete material;
	}
};

class Object
{
public:
	CUM::Transform transform;
	CUM::PrimitiveVector<Mesh>* meshVec;
	BBox bBox;
public:
	__duel__ Object(const CUM::Transform& _transform, CUM::PrimitiveVector<Mesh>* _meshVec)
		:transform(_transform), meshVec(_meshVec)
	{
		//To do...
		//bBox = mesh->GetBoundingBox();
	}
public:

	__host__ Object* copyToDevice()
	{
		Object objectInsWithDevicePtr(*this);

		CUM::PrimitiveVector<Mesh>* meshVecDevice = meshVec->copyToDevice();

		objectInsWithDevicePtr.meshVec = meshVecDevice;

		Object* objectDevice = CudaInsMemCpyHostToDevice(&objectInsWithDevicePtr);
		return objectDevice;
	}
	__duel__ void Release()
	{
		custd::OStream os;
		os << "Called Object::Release()!\n";
		meshVec->Release();
	}
	__duel__ const Bool HitTest(Ray& ray) const
	{
		return meshVec->HitTest(ray);
	}
private:
	//To do...
	__duel__ const Ray TransRay(const Ray& ray) const
	{
		Ray result;
		result.origin -= transform.translation;
		result.origin = CUM::applyInvQuaTransform(transform.rotation, result.origin);
		result.origin /= transform.scale;
		return Ray();
	}
};

//struct HierarchyTreeNode
//{
//	Object object;
//	custd::cuvector<HierarchyTreeNode*> childNodes;
//};

//class HierarchyTree
//{
//private:
//	static HierarchyTreeNode* headNod;
//public:
//	__duel__ const HierarchyTreeNode* GetInstance()
//	{
//		if (!headNod)
//			headNod = new HierarchyTreeNode;
//		return headNod;
//	}
//};


class Scene
{
public:
	PersCamera*  camera;
	CUM::PrimitiveVector<Object>* objectVec;

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

	__duel__ Scene(PersCamera* _camera, CUM::PrimitiveVector<Object>* _objectVec)
		:camera(_camera), objectVec(_objectVec)
	{
		
	}
public:
	__duel__ void EndRendering()
	{

	}

public:
	Scene* copyToDevice()
	{
		Scene sceneInsWithDevicePtr(*this);

		PersCamera* cameraDevice = camera->copyToDevice();
		CUM::PrimitiveVector<Object>* objectVecDevice = objectVec->copyToDevice();

		sceneInsWithDevicePtr.camera = cameraDevice;
		sceneInsWithDevicePtr.objectVec = objectVecDevice;

		Scene* sceneDevice = CudaInsMemCpyHostToDevice(&sceneInsWithDevicePtr);
		return sceneDevice;
	}
	__duel__ void Release()
	{
		custd::OStream os;
		os << "Called Scene::Release()!\n";
		CHECK(camera, "Scene::Release() error: camera can not be nullptr!");
		CHECK(objectVec, "Scene::Release() error: primitivesVectorPtr can not be nullptr!");
		if (camera)
		{
			camera->Release();
			camera = nullptr;
		}
		if (objectVec)
		{
			objectVec->Release();
			objectVec = nullptr;
		}
		
	}
public:
	__duel__ const CUM::Color3f GetSkyColor(const CUM::Vec3f& direction)
	{
		Float skyLerpFactor = CUM::max(direction.y,0.0);
		
		return CUM::Lerp(CUM::Color3f(1.0), CUM::Color3f(0.0, 0.0, 1.0), skyLerpFactor);
	}

};

__duel__ void Rendering(Scene* scene, Int globalIdx)
{
	if (globalIdx == 22911)
	{
		Int k = 4;
	}
	Scene& sceneDevice = *scene;
	PersCamera& camera = *scene->camera;
	CUM::PrimitiveVector<Object>& objectVec = *(scene->objectVec);
	CUM::Vec2i size = camera.renderTarget->size;

	Int length = size.x*size.y;
	const Int aliasingTime = 16;
	CUM::Vec2f deltaSampleUV = camera.renderTarget->deltaUV / aliasingTime;

	CUM::Vec2f uv;
	Int x, y;
	Float u, v;

	Ray ray;
	CUM::Color3f resultColor(0.0);
	CUM::Color3f sampledColor(1.0);
	CUM::Color3f tempColor(1.0);
	Ushort R, G, B, A;

	x = globalIdx % size.x;
	y = globalIdx / size.x;

	u = Float(x) / Float(size.x);
	v = Float(y) / Float(size.y);

	resultColor.r = 0.0;
	resultColor.g = 0.0;
	resultColor.b = 0.0;

	Int bounceTimeMinus1 = camera.bounceTime - 1; 

	for (Int i = 0; i < aliasingTime; i++)
	{
		for (Int j = 0; j < aliasingTime; j++)
		{
			uv.x = u + i * deltaSampleUV.x;
			uv.y = v + j * deltaSampleUV.y;

			ray = camera.GetRay(uv);

			sampledColor.r = 1.0;
			sampledColor.g = 1.0;
			sampledColor.b = 1.0;

			tempColor.r = 1.0;
			tempColor.g = 1.0;
			tempColor.b = 1.0;

			for (Int i = 0; i < camera.bounceTime; i++)
			{
				if (i == bounceTimeMinus1 && bounceTimeMinus1 != 0)
				{
					sampledColor = CUM::Color3f(0.0);
					resultColor = CUM::Color3f(0.0);
				}
				else if (objectVec.HitTest(ray))
				{
					tempColor = ray.record.sampledColor;
					//tempColor.r = ray.record.normal.x <= 0.0 ? 0.0 : ray.record.normal.x;
					//tempColor.g = ray.record.normal.y <= 0.0 ? 0.0 : ray.record.normal.y;
					//tempColor.b = ray.record.normal.z <= 0.0 ? 0.0 : ray.record.normal.z;
					//tempColor.r = 0.5*(ray.record.normal.x + 1.0);
					//tempColor.g = 0.5*(ray.record.normal.y + 1.0);
					//tempColor.b = 0.5*(ray.record.normal.z + 1.0);
					sampledColor *= tempColor;
					ray.CalculateNextRay();
				}
				else
				{
					sampledColor *= sceneDevice.GetSkyColor(ray.direction);
					break;
				}
			}
			resultColor += sampledColor;
		}
	}
	resultColor /= (aliasingTime*aliasingTime);
	resultColor *= 255.0;

	R = round(resultColor.r);
	G = round(resultColor.g);
	B = round(resultColor.b);
	A = round(255.0);
	camera.renderTarget->buffer[globalIdx].r = R;
	camera.renderTarget->buffer[globalIdx].g = G;
	camera.renderTarget->buffer[globalIdx].b = B;
	camera.renderTarget->buffer[globalIdx].a = A;

	//camera.renderTarget->buffer[globalIdx].r = 255;
	//camera.renderTarget->buffer[globalIdx].g = 0;
	//camera.renderTarget->buffer[globalIdx].b = 0;
	//camera.renderTarget->buffer[globalIdx].a = 255;
}

__global__ void RenderingOnDevice(Scene* scene)
{
	Int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;

	Rendering(scene, globalIdx);
}


__host__ void RenderingOnHost(Scene* scene)
{
	PersCamera& camera = *scene->camera;
	CUM::Vec2i size = camera.renderTarget->size;
	Int length = size.x*size.y;

	for (Int globalIdx = 0; globalIdx < length; globalIdx++)
	{
		Rendering(scene, globalIdx);
	}
}

#endif // !__GEOMETRY__CUH__
