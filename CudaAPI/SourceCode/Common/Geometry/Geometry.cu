#include "Geometry.cuh"
Texture* Camera::RenderTargetDevice = nullptr;

__duel__ void RenderingImplementation(Scene* scene, Int globalIdx)
{
	Scene& sceneDevice = *scene;
	PersCamera& camera = *scene->camera;
	CUM::PrimitiveVector<Object>& objectVec = *(scene->objectVec);
	CUM::Vec2i size = camera.renderTarget->size;

	Int length = size.x*size.y;
	const Int aliasingTime = 1;
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

	Int randOffsetIdx = globalIdx;

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
					tempColor = ray.record.sampledMaterial->Albedo;
					//tempColor.r = ray.record.normal.x <= 0.0 ? 0.0 : ray.record.normal.x;
					//tempColor.g = ray.record.normal.y <= 0.0 ? 0.0 : ray.record.normal.y;
					//tempColor.b = ray.record.normal.z <= 0.0 ? 0.0 : ray.record.normal.z;
					//tempColor.r = 0.5*(ray.record.normal.x + 1.0);
					//tempColor.g = 0.5*(ray.record.normal.y + 1.0);
					//tempColor.b = 0.5*(ray.record.normal.z + 1.0);
					sampledColor *= tempColor;

					ray.CalculateNextRay(randOffsetIdx);
					++randOffsetIdx;
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

	RenderingImplementation(scene, globalIdx);
}

__host__ void RenderingOnHost(Scene* scene)
{
	PersCamera& camera = *scene->camera;
	CUM::Vec2i size = camera.renderTarget->size;
	Int length = size.x*size.y;

	for (Int globalIdx = 0; globalIdx < length; globalIdx++)
	{
		RenderingImplementation(scene, globalIdx);
	}
}

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
