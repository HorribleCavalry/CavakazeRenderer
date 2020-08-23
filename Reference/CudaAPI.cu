#include "CudaAPI.cuh"

#pragma region CudaAPI

CUDA_API void OpenDebugConsole()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
}


bool haveInitializedRandNumber = false;
void generateRandData()
{
	std::default_random_engine randEngine(time(NULL));
	std::uniform_real_distribution<float> randGenerator(0.0, 1.0);

	cudaMemcpyToSymbol(device__randNumSize, &host_randNumSize, sizeof(int));

	float* host_randNumber_ptr = new float[host_randNumSize];
	for (int i = 0; i < host_randNumSize; i++)
	{
		host_randNumber_ptr[i] = randGenerator(randEngine);
	}

	cudaMalloc(&device_randNumber_ptr_pre, host_randNumSize * sizeof(float));
	cudaMemcpy(device_randNumber_ptr_pre, host_randNumber_ptr, host_randNumSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_randNumber_ptr, &device_randNumber_ptr_pre, sizeof(float*));


	cudaMemcpyToSymbol(device_randSamplerSize, &host_randSamplerSize, sizeof(int));

	Point2* host_randSampler_ptr = new Point2[host_randSamplerSize];
	for (int i = 0; i < host_randSamplerSize; i++)
	{
		host_randSampler_ptr[i].x = host_randNumber_ptr[2 * i];
		host_randSampler_ptr[i].y = host_randNumber_ptr[2 * i + 1];
	}

	cudaMalloc(&device_randSampler_ptr_pre, host_randSamplerSize * sizeof(Point2));
	cudaMemcpy(device_randSampler_ptr_pre, host_randSampler_ptr, host_randSamplerSize * sizeof(Point2), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_randSampler_ptr, &device_randSampler_ptr_pre, sizeof(Point2*));
	cudaError_t error = cudaGetLastError();


	cudaMemcpyToSymbol(device_randHemisphereVectorSize, &host_randHemisphereVectorSize, sizeof(int));

	Vec3* host_randHemisphereVector_ptr = new Vec3[host_randHemisphereVectorSize];
	for (int i = 0; i < host_randHemisphereVectorSize; i++)
	{
		float phi = host_randNumber_ptr[2 * i];
		float theta = acosf(host_randNumber_ptr[2 * i + 1]);
		host_randHemisphereVector_ptr[i].x = sinf(theta) * sinf(phi);
		host_randHemisphereVector_ptr[i].y = cosf(theta);
		host_randHemisphereVector_ptr[i].z = sinf(theta) * cosf(phi);
	}

	cudaMalloc(&device_randHemisphereVector_ptr_pre, host_randHemisphereVectorSize * sizeof(Vec3));
	cudaMemcpy(device_randHemisphereVector_ptr_pre, host_randHemisphereVector_ptr, host_randHemisphereVectorSize * sizeof(Vec3), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_randHemisphereVector_ptr, &device_randHemisphereVector_ptr_pre, sizeof(Vec3*));


	cudaMemcpyToSymbol(device_randSphereVectorSize, &host_randSphereVectorSize, sizeof(int));

	Vec3* host_randSphereVector_ptr = new Vec3[host_randSphereVectorSize];
	for (int i = 0; i < host_randSphereVectorSize; i++)
	{
		float phi = host_randNumber_ptr[2 * i];
		float theta = acosf(2 * host_randNumber_ptr[2 * i + 1] - 1.0f);
		host_randSphereVector_ptr[i].x = sinf(theta) * sinf(phi);
		host_randSphereVector_ptr[i].y = cosf(theta);
		host_randSphereVector_ptr[i].z = sinf(theta) * cosf(phi);
	}

	cudaMalloc(&device_randSphereVector_ptr_pre, host_randSphereVectorSize * sizeof(Vec3));
	cudaMemcpy(device_randSphereVector_ptr_pre, host_randSphereVector_ptr, host_randSphereVectorSize * sizeof(Vec3), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device_randSphereVector_ptr, &device_randSphereVector_ptr_pre, sizeof(Vec3*));


	delete[] host_randNumber_ptr;
	delete[] host_randSampler_ptr;
	delete[] host_randHemisphereVector_ptr;
	delete[] host_randSphereVector_ptr;
	haveInitializedRandNumber = true;
}
CUDA_API bool initializeResources(int _width, int _height, int _depth)
{

	bool isInitializedResourced = true;

	cudaMemcpyToSymbol(width, &_width, sizeof(int));
	cudaMemcpyToSymbol(height, &_height, sizeof(int));

	cudaMemcpyToSymbol(depth, &_depth, sizeof(int));

	element_num = _width * _height;

	frame_buffer_bytes = element_num * sizeof(BufferMap);

	cudaError_t error_gpu_f = cudaMalloc(&device_frame_buffer, frame_buffer_bytes);
	if (error_gpu_f != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Frame Buffer分配出错! %s\n", cudaGetErrorString(error_gpu_f));
		isInitializedResourced = false;
	}

	cudaError_t error_gpu_p = cudaMalloc(&device_picking_buffer, frame_buffer_bytes);
	if (error_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Picking Buffer分配出错! %s\n", cudaGetErrorString(error_gpu_f));
		isInitializedResourced = false;
	}

	if (!haveInitializedRandNumber)
	{
		generateRandData();
	}

	return isInitializedResourced;
}
CUDA_API int addBufferMap(BufferMap* bufferMap)
{
	bufferMaps.push_back(bufferMap);
	return bufferMaps.size() - 1;
}

CUDA_API bool freeResources()
{
	bool isFreedResourced = true;

	cudaError_t free_gpu_primitive = cudaFree(scene.device_primitive_list);
	if (free_gpu_primitive != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Primitives回收出错! %s\n", cudaGetErrorString(free_gpu_primitive));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_f = cudaFree(device_frame_buffer);
	if (free_gpu_f != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Frame Buffer回收出错! %s\n", cudaGetErrorString(free_gpu_f));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_p = cudaFree(device_picking_buffer);
	if (free_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Picking Buffer回收出错! %s\n", cudaGetErrorString(free_gpu_p));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_ranNumber = cudaFree(device_randNumber_ptr_pre);
	if (free_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Rand Number回收出错! %s\n", cudaGetErrorString(free_gpu_ranNumber));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_randSampler = cudaFree(device_randSampler_ptr_pre);
	if (free_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Rand Sampler回收出错! %s\n", cudaGetErrorString(free_gpu_randSampler));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_randHemisphere = cudaFree(device_randHemisphereVector_ptr_pre);
	if (free_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Rand Hemisphere回收出错! %s\n", cudaGetErrorString(free_gpu_randHemisphere));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_randSphere = cudaFree(device_randSphereVector_ptr_pre);
	if (free_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Rand Sphere回收出错! %s\n", cudaGetErrorString(free_gpu_randSphere));
		isFreedResourced = false;
	}

	cudaError_t free_gpu_texture_list = cudaFree(device_texture_list_pre);
	if (free_gpu_p != cudaError_t::cudaSuccess)
	{
		printf("CUDA error: GPU Texture List回收出错! %s\n", cudaGetErrorString(free_gpu_texture_list));
		isFreedResourced = false;
	}

	return isFreedResourced;
}
__global__ void testtexture(Texture texture)
{
	int length = texture.width*texture.height;
	for (int i = 0; i < length; i++)
	{
		printf("%d, %d, %d, %d\n", texture.data[4 * i], texture.data[4 * i + 1], texture.data[4 * i + 2], texture.data[4 * i + 3]);
	}
}
CUDA_API int addTexture(const char * path)
{
	std::cout << path << std::endl;
	int texture_width, texture_height, nrChannels;
	unsigned char *data = stbi_load(path, &texture_width, &texture_height, &nrChannels, 0);

	int eleNum = texture_width * texture_height;
	int texture_bytes = eleNum * 4 * sizeof(unsigned char);

	std::cout << texture_width<<"\t"<< texture_height << std::endl;


	Texture texture;

	texture.width = texture_width;
	texture.height = texture_height;
	texture.nrChannel = nrChannels;

	cudaMalloc(&texture.data, texture_bytes);
	cudaError_t error = cudaMemcpy(texture.data, data, texture_bytes, cudaMemcpyHostToDevice);

	if (error!=cudaError_t::cudaSuccess)
	{
		std::cout <<"在GPU中分配纹理空间失败: "<< cudaGetErrorString(error) << std::endl;
	}

	stbi_image_free(data);
	host_texture_list.push_back(texture);

	return host_texture_list.size() - 1;
}

CUDA_API bool deleteTexture(int index)
{
	//CHECK(cudaFree(host_texture_list[index].data))
	host_texture_list.erase(host_texture_list.begin() + index);

	return true;
}

__global__ void ChangeTexture(int index, int _width, int _height, int _nrChannel, unsigned char* data)
{
	device_texture_list[index].width = _width;
	device_texture_list[index].height = _height;
	device_texture_list[index].nrChannel = _nrChannel;
	device_texture_list[index].data = data;
}
CUDA_API bool changeTexture(int index, const char * path)
{
	//int texture_width, texture_height, nrChannels;
	//unsigned char *data = stbi_load(path, &texture_width, &texture_height, &nrChannels, 0);

	//int eleNum = texture_width * texture_height;
	//int texture_bytes = eleNum * 4 * sizeof(unsigned char);

	//cudaMalloc(&host_texture_list[index].data, texture_bytes);
	//cudaMemcpy(texture.data, data, texture_bytes, cudaMemcpyHostToDevice);

	//ChangeTexture << <1, 1 >> > (index, texture_width, texture_height, nrChannels, host_texture_list[index].data);


	//stbi_image_free(data);

	return true;
}

CUDA_API bool generateTextureList()
{
	int textureNum = host_texture_list.size();
	Texture* texture_list = new Texture[textureNum];
	for (int i = 0; i < textureNum; i++)
	{
		texture_list[i] = host_texture_list[i];
	}
	cudaMalloc(&device_texture_list_pre, textureNum * sizeof(Texture));
	cudaError_t texture_list_error =  cudaMemcpy(device_texture_list_pre, texture_list, textureNum * sizeof(Texture), cudaMemcpyHostToDevice);
	if (texture_list_error!= cudaError_t::cudaSuccess)
	{
	std::cout << "纹理列表生成出错: " << cudaGetErrorString(texture_list_error);
	}
	cudaMemcpyToSymbol(device_texture_list, &device_texture_list_pre, sizeof(Texture*));

	return true;
}

CUDA_API int AddSphere(float c_x, float c_y, float c_z, float r, float g, float b, float a, float radius)
{
	scene.primitives_num++;
	Primitive sphere(Primitive_type::Sphere, Point3(c_x, c_y, c_z), Color(r, g, b, a), radius);
	primitives.push_back(sphere);
	return primitives.size() - 1;
}

CUDA_API int AddTriangle(float r, float g, float b, float a, float point0_x, float point0_y, float point0_z, float point1_x, float point1_y, float point1_z, float point2_x, float point2_y, float point2_z, float uv0_u, float uv0_v, float uv1_u, float uv1_v, float uv2_u, float uv2_v)
{
	scene.primitives_num++;
	Primitive triangle(Primitive_type::Triangle, Color(r, g, b, a),
		Point3(point0_x, point0_y, point0_z), Point3(point1_x, point1_y, point1_z), Point3(point2_x, point2_y, point2_z),
		Point2(uv0_u, uv0_v), Point2(uv1_u, uv1_v), Point2(uv2_u, uv2_v));
	primitives.push_back(triangle);
	return primitives.size() - 1;
}

CUDA_API int AddPlane(float c_x, float c_y, float c_z, float r, float g, float b, float a, float normal_x, float normal_y, float normal_z)
{
	scene.primitives_num++;
	Primitive plane(Plane, Point3(c_x, c_y, c_z), Color(r, g, b, a), Normal(normal_x, normal_y, normal_z));
	primitives.push_back(plane);
	return primitives.size() - 1;
}

CUDA_API bool SendCamera(float position_x, float position_y, float position_z, float rotation_x, float rotation_y, float rotation_z, float rotation_w, float fov, float aspectRatio, float viewDistance, int sampler)
{
	scene.camera = Camera(Point3(position_x, position_y, position_z), rotation_x, rotation_y, rotation_z, rotation_w, fov, aspectRatio, viewDistance, (Sampler)sampler);
	return true;
}

CUDA_API bool deletePrimitive(int index)
{
	bool isDeletePrimitive = true;

	scene.primitives_num--;
	primitives.erase(primitives.begin() + index);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isDeletePrimitive = false;
		printf("CUDA error: 删除图元出错！%s\n", cudaGetErrorString(error));
	}
	return isDeletePrimitive;
}

CUDA_API bool changePrimitive(int index, Primitive_type type, float c_x, float c_y, float c_z, float r, float g, float b, float a, float n_x, float n_y, float n_z, float radius, float p0_x, float p0_y, float p0_z, float p1_x, float p1_y, float p1_z, float p2_x, float p2_y, float p2_z, float uv0_u, float uv0_v, float uv1_u, float uv1_v, float uv2_u, float uv2_v)
{
	bool isChangePrimitive = true;
	Point3 centre(c_x, c_y, c_z);
	Color materialColor(r, g, b, a);
	Normal normal(n_x, n_y, n_z);
	Point3 points[3];
	points[0] = Point3(p0_x, p0_y, p0_z);
	points[1] = Point3(p1_x, p1_y, p1_z);
	points[2] = Point3(p2_x, p2_y, p2_z);
	Point2 uv[3];
	uv[0] = Point2(uv0_u, uv0_v);
	uv[1] = Point2(uv1_u, uv1_v);
	uv[2] = Point2(uv2_u, uv2_v);

	Primitive changed(
		type,
		centre,
		materialColor,
		normal,
		radius,
		points[0],
		points[1],
		points[2],
		uv[0],
		uv[1],
		uv[2]
	);

	primitives[index] = changed;
	scene.change_device_scene(index, changed);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitive = false;
		printf("CUDA error: 修改图元出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitive;
}

__global__ void ChangePrimitiveType(Primitive* device_primitive_list, int index, Primitive_type type)
{
	device_primitive_list[index].type = type;
}
CUDA_API bool changePrimitiveType(int index, Primitive_type type)
{
	bool isChangePrimitiveTypeSuccess = true;
	primitives[index].type = type;
	ChangePrimitiveType << <1, 1 >> > (scene.device_primitive_list, index, type);
	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitiveTypeSuccess = false;
		printf("CUDA error: 修改图元Type出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitiveTypeSuccess;
}

__global__ void ChangePrimitiveCentre(Primitive* device_primitive_list, int index, Point3 centre)
{
	device_primitive_list[index].centre = centre;
}
CUDA_API bool changePrimitiveCentre(int index, float c_x, float c_y, float c_z)
{
	bool isChangePrimitiveCentre = true;
	Point3 centre(c_x, c_y, c_z);
	primitives[index].centre = centre;
	ChangePrimitiveCentre << <1, 1 >> > (scene.device_primitive_list, index, centre);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitiveCentre = false;
		printf("CUDA error: 修改图元Centre出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitiveCentre;
}

__global__ void ChangePrimitiveColor(Primitive* device_primitive_list, int index, Color materialColor)
{
	device_primitive_list[index].materialColor = materialColor;
}
CUDA_API bool changePrimitiveColor(int index, float r, float g, float b, float a)
{
	bool isChangePrimitiveColor = true;

	Color materialColor(r, g, b, a);
	primitives[index].materialColor = materialColor;
	ChangePrimitiveColor << <1, 1 >> > (scene.device_primitive_list, index, materialColor);
	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitiveColor = false;
		printf("CUDA error: 修改图元Color出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitiveColor;
}

__global__ void ChangePrimitiveNormal(Primitive* device_primitive_list, int index, Normal normal)
{
	device_primitive_list[index].normal = normal;
}
CUDA_API bool changePrimitiveNormal(int index, float x, float y, float z)
{
	bool isChangePrimitiveNormal = true;

	Normal normal(x, y, z);
	primitives[index].normal = normal;
	ChangePrimitiveNormal << <1, 1 >> > (scene.device_primitive_list, index, normal);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitiveNormal = false;
		printf("CUDA error: 修改图元Normal出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitiveNormal;
}

__global__ void ChangePrimitiveRadius(Primitive* device_primitive_list, int index, float radius)
{
	device_primitive_list[index].radius = radius;
}
CUDA_API bool changePrimitiveRadius(int index, float radius)
{
	bool isChangePrimitiveRadius = true;

	primitives[index].radius = radius;
	ChangePrimitiveRadius << <1, 1 >> > (scene.device_primitive_list, index, radius);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitiveRadius = false;
		printf("CUDA error: 修改图元Radius出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitiveRadius;
}

__global__ void ChangePrimitivePoints(Primitive* device_primitive_list, int index, Point3 p0, Point3 p1, Point3 p2)
{
	device_primitive_list[index].points[0] = p0;
	device_primitive_list[index].points[1] = p1;
	device_primitive_list[index].points[2] = p2;
	device_primitive_list[index].updateNormal();
}
CUDA_API bool changePrimitivePoints(int index, float p0_x, float p0_y, float p0_z, float p1_x, float p1_y, float p1_z, float p2_x, float p2_y, float p2_z)
{
	bool isChangePrimitivePoints = true;

	Point3 points[3];
	points[0].x = p0_x; points[0].y = p0_y; points[0].z = p0_z;
	points[1].x = p1_x; points[1].y = p1_y; points[1].z = p1_z;
	points[2].x = p2_x; points[2].y = p2_y; points[2].z = p2_z;

	primitives[index].points[0] = points[0];
	primitives[index].points[1] = points[1];
	primitives[index].points[2] = points[2];

	primitives[index].updateNormal();

	ChangePrimitivePoints << <1, 1 >> > (scene.device_primitive_list, index, points[0], points[1], points[2]);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitivePoints = false;
		printf("CUDA error: 修改图元Points出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitivePoints;

}

__global__ void ChangePrimitiveUVs(Primitive* device_primitive_list, int index, Point2 uv0, Point2 uv1, Point2 uv2)
{
	device_primitive_list[index].uv[0] = uv0;
	device_primitive_list[index].uv[1] = uv1;
	device_primitive_list[index].uv[2] = uv2;
}
CUDA_API bool changePrimitiveUV(int index, float uv0_u, float uv0_v, float uv1_u, float uv1_v, float uv2_u, float uv2_v)
{
	bool isChangePrimitiveUVs = true;

	Point2 uv[3];
	uv[0].x = uv0_u; uv[0].y = uv0_v;
	uv[1].x = uv1_u; uv[1].y = uv1_v;
	uv[2].x = uv2_u; uv[2].y = uv2_v;

	primitives[index].uv[0] = uv[0];
	primitives[index].uv[1] = uv[1];
	primitives[index].uv[2] = uv[2];

	ChangePrimitiveUVs << <1, 1 >> > (scene.device_primitive_list, index, uv[0], uv[1], uv[2]);

	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isChangePrimitiveUVs = false;
		printf("CUDA error: 修改图元Points出错！%s\n", cudaGetErrorString(error));
	}
	return isChangePrimitiveUVs;
}

CUDA_API Primitive_type checkPrimitiveType(int index)
{
	return primitives[index].type;
}

CUDA_API float checkPrimitiveCentre(int index, int c_idx)
{
	float result;
	switch (c_idx)
	{
	case 0:result = primitives[index].centre.x; break;
	case 1:result = primitives[index].centre.y; break;
	case 2:result = primitives[index].centre.z; break;
	}
	return result;
}

CUDA_API float checkPrimitiveColor(int index, int c_idx)
{
	float result;
	switch (c_idx)
	{
	case 0:result = primitives[index].materialColor.r;  break;
	case 1:result = primitives[index].materialColor.g; break;
	case 2:result = primitives[index].materialColor.b; break;
	case 3:result = primitives[index].materialColor.a; break;
	}
	return result;
}

CUDA_API float checkPrimitiveNormal(int index, int n_idx)
{
	float result;
	switch (n_idx)
	{
	case 0:result = primitives[index].normal.x; break;
	case 1:result = primitives[index].normal.y; break;
	case 2:result = primitives[index].normal.z; break;
	}
	return result;
}

CUDA_API float checkPrimitiveRadius(int index)
{
	return primitives[index].radius;
}

CUDA_API float checkPrimitivePoints(int index, int p_idx)
{
	float result;
	switch (p_idx)
	{
	case 0:result = primitives[index].points[0].x; break;
	case 1:result = primitives[index].points[0].y; break;
	case 2:result = primitives[index].points[0].z; break;

	case 3:result = primitives[index].points[1].x; break;
	case 4:result = primitives[index].points[1].y; break;
	case 5:result = primitives[index].points[1].z; break;

	case 6:result = primitives[index].points[2].x; break;
	case 7:result = primitives[index].points[2].y; break;
	case 8:result = primitives[index].points[2].z; break;
	}
	return result;
}

CUDA_API float checkPrimitiveUV(int index, int u_idx)
{
	float result;
	switch (u_idx)
	{
	case 0:result = primitives[index].uv[0].x; break;
	case 1:result = primitives[index].uv[0].y; break;

	case 2:result = primitives[index].uv[1].x; break;
	case 3:result = primitives[index].uv[1].y; break;

	case 4:result = primitives[index].uv[2].x; break;
	case 5:result = primitives[index].uv[2].y; break;
	}
	return result;
}

CUDA_API bool setPrimitiveDesinyPBRMaterial(int _primitive_index, int _Albedo_index, int _Normal_index, int _Metallic_index, int _Roughness_index, int _AO_index)
{
	primitives[_primitive_index].material.brdf = BRDF::DisneyPBR;
	primitives[_primitive_index].material.Albedo_index = _Albedo_index;
	primitives[_primitive_index].material.Normal_index = _Normal_index;
	primitives[_primitive_index].material.Metallic_index = _Metallic_index;
	primitives[_primitive_index].material.Roughness_index = _Roughness_index;
	primitives[_primitive_index].material.AO_index = _AO_index;
	return true;
}

CUDA_API bool setPrimitiveBlinPhongMaterial(int _primitive_index, int _Albedo_index, int _Normal_index, int _Roughness_index, int _AO_index)
{
	primitives[_primitive_index].material.brdf = BRDF::BlinPhong;
	primitives[_primitive_index].material.Albedo_index = _Albedo_index;
	primitives[_primitive_index].material.Normal_index = _Normal_index;
	primitives[_primitive_index].material.Roughness_index = _Roughness_index;
	primitives[_primitive_index].material.AO_index = _AO_index;
	return true;
}

CUDA_API bool generateScene()
{
	bool isGenerateSceneSuccess = true;
	if (!scene.generate_device_scene())
	{
		isGenerateSceneSuccess = false;
	}
	return isGenerateSceneSuccess;
}

__global__ void RenderScene(BufferMap* device_frame_buffer, BufferMap* device_picking_buffer, Scene _scene)
{
	int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;


	int x = globalIdx % width;
	int y = -globalIdx / width + height;

	float u = (float)x / (float)width;
	float v = (float)y / (float)height;
	Ray ray = _scene.camera.getRay(u, v);

	Color result_color(1.0f, 1.0f, 1.0f, 1.0f);
	Color tempColor(1.0f, 1.0f, 1.0f, 1.0f);
	bool haveHitPrimitives = false;

	int picking_index = -1;
	for (int i = 0; i < depth; i++)//遍历n次深度
	{
		haveHitPrimitives = false;//在这一次的图元遍历中是否hit到过primitive
		if (_scene.HitTest(ray))//在primitive.hit()中遍历所有的图元与ray求交，找到交点并将ray修改为下一次光线弹射的方向,没有交点ray就不改动
		{			
			tempColor = _scene.GetHitColor(ray);

			if (i == 0)
			{
				picking_index = ray.record.primitiveIndex;
			}

			ray.record.primitiveIndex = -1;//当获取到tempColor后，这个ray就回归至初始状态，把Index值成-1
			haveHitPrimitives = true;//射线成功的和一位primitive有了交点
		}

		if (!haveHitPrimitives)
		{
			result_color *= Color::GetBackgroundColor(ray.direction.y);//如果遍历了所有的图元还是没任何交点，那么就干脆乘一下天空的颜色
			break;
		}
		else if (i == depth - 1 && depth != 1)
		{
			result_color = Color(0.0f, 0.0f, 0.0f, 1.0f);
		}
		else
		{
			result_color *= tempColor;
		}
	}

	result_color.toneMapping();
	result_color.transToGamma();
	result_color *= 255.0f;

	device_frame_buffer[globalIdx].r = (unsigned char)roundf(result_color.r);
	device_frame_buffer[globalIdx].g = (unsigned char)roundf(result_color.g);
	device_frame_buffer[globalIdx].b = (unsigned char)roundf(result_color.b);

	device_picking_buffer[globalIdx].r = (unsigned char)(picking_index / 256 * 256);
	device_picking_buffer[globalIdx].g = (unsigned char)(picking_index / 256 % 256);
	device_picking_buffer[globalIdx].b = (unsigned char)(picking_index % 256);
}

CUDA_API bool renderScene(int frame_buffer_index, int picking_buffer_index)
{

	bool isRenderSuccess = true;

	int threads_num_per_block = 32;
	int block_num = element_num / threads_num_per_block;

	RenderScene << <block_num, threads_num_per_block >> > (device_frame_buffer, device_picking_buffer, scene);
	cudaError_t error = cudaGetLastError();
	if (error != cudaError_t::cudaSuccess)
	{
		isRenderSuccess = false;
		printf("CUDA error: 渲染出错！%s\n", cudaGetErrorString(error));
	}

	cudaError_t error_cpy_F = cudaMemcpy(bufferMaps[frame_buffer_index], device_frame_buffer, frame_buffer_bytes, cudaMemcpyDeviceToHost);
	if (error_cpy_F != cudaError_t::cudaSuccess)
	{
		isRenderSuccess = false;
		printf("CUDA error: Device To Hos Cpy出错！%s\n", cudaGetErrorString(error_cpy_F));
	}

	cudaError_t error_cpy_P = cudaMemcpy(bufferMaps[picking_buffer_index], device_frame_buffer, frame_buffer_bytes, cudaMemcpyDeviceToHost);
	if (error_cpy_P != cudaError_t::cudaSuccess)
	{
		isRenderSuccess = false;
		printf("CUDA error: Device To Hos Cpy出错！%s\n", cudaGetErrorString(error_cpy_P));
	}

	//cudaDeviceSynchronize();

	return isRenderSuccess;
}

__global__ void test1()
{
	printf("%d\n",depth);
}
CUDA_API void test()
{
}

#pragma endregion

#pragma test
int main()
{
	//int width, height, nrChannels;
	//unsigned char *data = stbi_load("C:\\Users\\Hordr\\source\\repos\\CudaAPI\\x64\\Debug\\Textures\\test.png", &width, &height, &nrChannels, 0);
	////std::cout << (int)data[5] << std::endl;
	////std::cout << nrChannels << std::endl;
	////int temp = sizeof(unsigned char);
	////std::cout << sizeof(data) / sizeof(unsigned char) << std::endl;
	//int elenum = width * height;
	//for (int i = 0; i < elenum; i++)
	//{
	//	std::cout << (int)data[4 * i + 0] << "\t";
	//	std::cout << (int)data[4 * i + 1] << "\t";
	//	std::cout << (int)data[4 * i + 2] << "\t";
	//	std::cout << (int)data[4 * i + 3] << std::endl;
	//}
	//stbi_image_free(data);
	//std::cout << "tesing..." << std::endl;

	//BufferMap* fb = new BufferMap[256*144];
	//BufferMap* pb = new BufferMap[256 * 144];
	initializeResources(256, 144, 100);
	test1 << <1, 1 >> > ();
	//addBufferMap(fb);
	//addBufferMap(pb);
	////AddSphere(0.0f, 0.0f, -5.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f);
	//AddPlane(0.0f, -2.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f);
	//addTexture("C:\\Users\\Hordr\\source\\repos\\CavakazeRenderer\\CavakazeRenderer\\bin\\Debug\\Textures\\albedo.png");
	//addTexture("C:\\Users\\Hordr\\source\\repos\\CavakazeRenderer\\CavakazeRenderer\\bin\\Debug\\Textures\\normal.png");
	//addTexture("C:\\Users\\Hordr\\source\\repos\\CavakazeRenderer\\CavakazeRenderer\\bin\\Debug\\Textures\\metallic.png");
	//addTexture("C:\\Users\\Hordr\\source\\repos\\CavakazeRenderer\\CavakazeRenderer\\bin\\Debug\\Textures\\roughness.png");
	//addTexture("C:\\Users\\Hordr\\source\\repos\\CavakazeRenderer\\CavakazeRenderer\\bin\\Debug\\Textures\\ao.png");
	//generateTextureList();
	//setPrimitiveDesinyPBRMaterial(0, 0, 1, 2, 3, 4);
	//generateScene();
	//SendCamera(0.0f, 0.0f, 0.0f, -0.008749789f, 0.7140866f, -0.699945867f, -0.008926612f, 1.04719341f, 1.77777779f, 5, 0);
	//renderScene(0,1);
}

#pragma endregion
