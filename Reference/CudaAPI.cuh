#pragma once

#define CUDA_API __declspec(dllexport)
#include "CudaUtil.cuh"

#pragma region image relevant global variable
std::vector<BufferMap*> bufferMaps;
BufferMap* device_frame_buffer = 0;
BufferMap* device_picking_buffer = 0;

int element_num = 0;
int frame_buffer_bytes = 0;
#pragma endregion

#pragma region texture relevant global variables
std::vector<Texture> host_texture_list;
#pragma endregion

#pragma region primitive relevant global variable
int primitives_num = 0;
int primitives_byte = 0;
#pragma endregion

#pragma region scene relevant global variable
Scene scene;
#pragma endregion

extern "C"
{
#pragma region CudaAPI

#pragma region Console
	CUDA_API void OpenDebugConsole();
#pragma endregion

#pragma region Malloc&Free
	CUDA_API bool initializeResources(int _width, int _height, int _depth);

	CUDA_API int addBufferMap(BufferMap* bufferMap);

	CUDA_API bool freeResources();

#pragma endregion

#pragma region texture add delete change generate
	CUDA_API int addTexture(const char* path);

	CUDA_API bool deleteTexture(int index);

	CUDA_API bool changeTexture(int index, const char* path);

	CUDA_API bool generateTextureList();

#pragma endregion

#pragma region primitive add delete change check
	CUDA_API int AddSphere(
		float c_x, float c_y, float c_z,//centre
		float r, float g, float b, float a,//materialColor
		float radius);//radius

	CUDA_API int AddTriangle(
		float r, float g, float b, float a,//materialColor
		float point0_x, float point0_y, float point0_z,//point0
		float point1_x, float point1_y, float point1_z,//point1
		float point2_x, float point2_y, float point2_z,//point2
		float uv0_u, float uv0_v,//uv0
		float uv1_u, float uv1_v,//uv1
		float uv2_u, float uv2_v//uv2
	);

	CUDA_API int AddPlane(
		float c_x, float c_y, float c_z,//centre
		float r, float g, float b, float a,//materialColor
		float normal_x, float normal_y, float normal_z);//normal

	CUDA_API bool SendCamera(
		float position_x, float position_y, float position_z,
		float rotation_x, float rotation_y, float rotation_z, float rotation_w,
		float fov, float aspectRatio, float viewDistance, int sampler);

	CUDA_API bool deletePrimitive(int index);

	CUDA_API bool changePrimitive(
		int index,
		Primitive_type type,
		float c_x, float c_y, float c_z,
		float r, float g, float b, float a,
		float n_x, float n_y, float n_z,
		float radius,
		float p0_x, float p0_y, float p0_z,
		float p1_x, float p1_y, float p1_z,
		float p2_x, float p2_y, float p2_z,
		float uv0_u, float uv0_v,
		float uv1_u, float uv1_v,
		float uv2_u, float uv2_v
	);

	CUDA_API bool changePrimitiveType(int index, Primitive_type type);
	CUDA_API bool changePrimitiveCentre(int index, float c_x, float c_y, float c_z);
	CUDA_API bool changePrimitiveColor(int index, float r, float g, float b, float a);
	CUDA_API bool changePrimitiveNormal(int index, float x, float y, float z);
	CUDA_API bool changePrimitiveRadius(int index, float radius);
	CUDA_API bool changePrimitivePoints(int index,
		float p0_x, float p0_y, float p0_z,
		float p1_x, float p1_y, float p1_z,
		float p2_x, float p2_y, float p2_z
	);
	CUDA_API bool changePrimitiveUV(int index,
		float uv0_u, float uv0_v,
		float uv1_u, float uv1_v,
		float uv2_u, float uv2_v
	);


	CUDA_API Primitive_type checkPrimitiveType(int index);
	CUDA_API float checkPrimitiveCentre(int index, int c_idx);
	CUDA_API float checkPrimitiveColor(int index, int c_idx);
	CUDA_API float checkPrimitiveNormal(int index, int n_idx);
	CUDA_API float checkPrimitiveRadius(int index);
	CUDA_API float checkPrimitivePoints(int index, int p_idx);
	CUDA_API float checkPrimitiveUV(int index, int u_idx);

#pragma endregion

#pragma region primitive material
	CUDA_API bool setPrimitiveBlinPhongMaterial(int _primitive_index, int _Albedo_index, int _Normal_index, int _Roughness_index, int _AO_index);
	CUDA_API bool setPrimitiveDesinyPBRMaterial(int _primitive_index, int _Albedo_index, int _Normal_index, int _Metallic_index, int _Roughness_index, int _AO_index);

	//CUDA_API bool changePrimitiveMaterialDesinyPBR(int _primitive_index);

	//CUDA_API bool changePrimitiveMaterialDesinyPBR(int _primitive_index);
#pragma endregion

#pragma region generate&render
	CUDA_API bool generateScene();

	CUDA_API bool renderScene(int frame_buffer_index, int picking_buffer_index);
#pragma endregion

#pragma region Debug&Test
	CUDA_API void test();
#pragma endregion
}