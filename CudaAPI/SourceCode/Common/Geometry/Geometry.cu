﻿#include "Geometry.cuh"
#include "../../CudaSTD/cuvector.cuh"
Texture* Camera::RenderTargetDevice = nullptr;

#ifdef RUN_ON_DEVICE
__device__
#endif // RUN_ON_DEVICE
#ifdef RUN_ON_HOST
__host__
#endif // RUN_ON_HOST
void RenderingImplementation(Scene* scene, Int globalIdx)
{
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
	Ushort R, G, B, A;

	x = globalIdx % size.x;
	y = globalIdx / size.x;

	u = Float(x) / Float(size.x);
	v = Float(y) / Float(size.y);

	resultColor.r = 0.0;
	resultColor.g = 0.0;
	resultColor.b = 0.0;

	Int bounceTimeMinus1 = camera.bounceTime - 1;

	custd::cuvector<CUM::Color3f> ColorList;
	custd::cuvector<CUM::Color3f> LightRadianceList;

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

			ColorList.Release();
			LightRadianceList.Release();
			for (Int i = 0; i < camera.bounceTime; i++)
			{

				if (objectVec.HitTest(ray))
				{
					ray.ProcessSampledResult();
					ColorList.push_back(ray.record.sampledColor);
					LightRadianceList.push_back(ray.record.sampledLightRadiance);
				}
				else
				{
					ColorList.push_back(sceneDevice.GetSkyColor(ray.direction));
					break;
				}
			}

			sampledColor = CalculateSampledColor(&ColorList, &LightRadianceList);

			resultColor += sampledColor;
		}
	}
	const Float p = 1.0 / 2.2;
	resultColor /= (aliasingTime*aliasingTime);
	resultColor.r = pow(resultColor.r, p);
	resultColor.g = pow(resultColor.g, p);
	resultColor.b = pow(resultColor.b, p);
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

#ifdef RUN_ON_DEVICE
__global__ void RenderingOnDevice(Scene* scene)
{
	Int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;

	RenderingImplementation(scene, globalIdx);
}

//__global__ void Median
#endif // RUN_ON_DEVICE

#ifdef RUN_ON_HOST
void RenderingOnHost(Scene* scene)
{
	PersCamera& camera = *scene->camera;
	CUM::Vec2i size = camera.renderTarget->size;
	Int length = size.x*size.y;

	for (Int globalIdx = 0; globalIdx < length; globalIdx++)
	{
		RenderingImplementation(scene, globalIdx);
	}
}
#endif // RUN_ON_HOST


__host__ void Rendering(Scene* sceneHost, Scene* sceneDevice, Int imageLength)
{
#ifdef RUN_ON_DEVICE
	Int threadNum = 32;
	Int blockNum = imageLength / threadNum;
	RenderingOnDevice << <blockNum, threadNum >> > (sceneDevice);
	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	sceneHost->camera->renderTarget->CopyFromDevice(PersCamera::RenderTargetDevice);
#endif // RUN_ON_DEVICE

#ifdef RUN_ON_HOST
	RenderingOnHost(sceneHost);
#endif // RUN_ON_HOST

	//if (isOnDevice)
	//{
	//	Int threadNum = 32;
	//	Int blockNum = imageLength / threadNum;
	//	RenderingOnDevice << <blockNum, threadNum >> > (sceneDevice);
	//	cudaError_t error = cudaGetLastError();
	//	if (error != cudaError_t::cudaSuccess)
	//	{
	//		printf("%s\n", cudaGetErrorString(error));
	//	}
	//	sceneHost->camera->renderTarget->CopyFromDevice(PersCamera::RenderTargetDevice);
	//}
	//else
	//{
	//	RenderingOnHost(sceneHost);
	//}
}
