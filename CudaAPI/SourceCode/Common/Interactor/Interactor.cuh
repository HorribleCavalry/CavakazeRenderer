#ifndef __INTERACTOR__CUH__
#define __INTERACTOR__CUH__

#include "../CudaPrimitivesVector.cuh"

class RayProcessor
{
public:
#ifdef RUN_ON_DEVICE
	__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
	__host__
#endif // RUN_ON_HOST
	void Processing(CUM::PrimitiveVector<Object>& objects, Ray& ray)
	{
		//lightVec.Sampling(*this);
		const CUM::Vec3f L = CUM::normalize(CUM::Vec3f(1.0, 1.0, -1.0));
		CUM::Vec3f N(ray.record.normal.x, ray.record.normal.y, ray.record.normal.z);
		CUM::Vec3f V(ray.direction);
		CUM::Vec3f H(CUM::normalize(L + V));
		CUM::Color3f LightRadiance(1.0, 0.8, 0.25);
		CHECK(ray.record.sampledMaterial, "Ray::InteractWithSampledResultAndShadingFromLight() error: the sampledMaterial can not be nullptr!");
		Float distance = 1.0;

		Ray shadowRay;
		shadowRay.origin = ray.record.hitPoint;
		shadowRay.direction = L;

		Bool isHit = objects.HitTest(shadowRay);
		CUM::Color3f shadedLightRadience;
		if (isHit)
		{
			shadedLightRadience = CUM::Color3f(0.0);
		}
		else
		{
			shadedLightRadience = ray.record.sampledMaterial->ShadeWithDirectLight(N, V, H, L, 1.0, LightRadiance);
		}

		ray.record.sampledLightRadiance = shadedLightRadience;
		ray.record.sampledColor = ray.record.sampledMaterial->Albedo;

		ray.direction = ray.record.sampledMaterial->InteractWithRay(N, V);
		ray.origin = ray.record.hitPoint;
	}
};

#endif // !__INTERACTOR__CUH__
