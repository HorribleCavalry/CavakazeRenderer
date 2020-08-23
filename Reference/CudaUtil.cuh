#ifndef __CUDAUTIL__CUH__
#define __CUDAUTIL__CUH__

//#define CHECK(call)
//{
//	const cudaError_t error = call;
//	if (error != cudaSuccess)
//	{
//		printf("CUDA Error: %s: %d, ", __FILE__, __LINE__);
//		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
//		exit(1);
//	}
//}
#define PI 3.14159265359f
#include <time.h>
#include <vector>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <random>
#include <iostream>
#include <algorithm>
#include "stb_image.h"
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region class declaration
class Ray;
class Vec2;
class Vec3;
class Color;
class Scene;
class Point2;
class Point3;
class Record;
class Normal;
class Texture;
class Camera;
class Material;
class Primitive;
class Quaternion;
#pragma endregion


#pragma region random numbers
int host_randNumSize = 1024;
__device__ int device__randNumSize;

float* device_randNumber_ptr_pre;
__device__ float* device_randNumber_ptr;


int host_randSamplerSize = host_randNumSize / 2;
__device__ int device_randSamplerSize;

Point2* device_randSampler_ptr_pre;
__device__ Point2* device_randSampler_ptr;


int host_randHemisphereVectorSize = host_randNumSize / 2;
__device__ int device_randHemisphereVectorSize;

Vec3* device_randHemisphereVector_ptr_pre;
__device__ Vec3* device_randHemisphereVector_ptr;


int host_randSphereVectorSize = host_randNumSize / 2;
__device__ int device_randSphereVectorSize;

Vec3* device_randSphereVector_ptr_pre;
__device__ Vec3* device_randSphereVector_ptr;

#pragma endregion

#pragma region primitive list
std::vector <Primitive>primitives;

#pragma endregion


#pragma region texture device global list
Texture* device_texture_list_pre;
__device__ Texture* device_texture_list;

#pragma endregion

#pragma region common device static global variable
__device__ int width = 0;
__device__ int height = 0;
__device__ int depth;
#pragma endregion
#pragma region common operator algorithm
inline __host__ __device__ Vec3 operator*(const float & t, const Vec3 & v);

inline __host__ __device__ Vec3 operator-(const Point3& p1, const Point3& p2);
inline __host__ __device__ Point3 operator+(const Point3& p, const Vec3& v);
inline __host__ __device__ Point3 operator+(const Point3& p1, const Point3& p2);
inline __host__ __device__ Point3 operator-(const Point3& p, const Vec3& v);

inline __host__ __device__ Color operator+(const Color& c1, const Color& c2);
inline __host__ __device__ Color operator+(const Color& c, const float& t);
inline __host__ __device__ Color operator+(const float& t, const Color& c);
inline __host__ __device__ Color operator-(const Color& c1, const Color& c2);
inline __host__ __device__ Color operator-(const Color& c, const float& t);
inline __host__ __device__ Color operator-(const float& t, const Color& c);
inline __host__ __device__ Color operator*(const Color& c1, const Color& c2);
inline __host__ __device__ Color operator*(const float& t, const Color& c);
inline __host__ __device__ Color operator*(const Color& c, const float& t);
inline __host__ __device__ Color operator/(const Color& c1, const Color& c2);
inline __host__ __device__ Color operator/(const Color& c, const float& t);
inline __host__ __device__ Color mix(const Color& c1, const Color& c2, float t);
inline __host__ __device__ float mix(const float& t1, const float& t2, float t);

#pragma endregion

#pragma region class definition

class Vec2
{
public:
	float x, y;
	Vec2();
	Vec2(float _x, float _y);
	Vec2(const Vec2& v);
	~Vec2();

private:

};

class Vec3
{
public:
	float x, y, z;
	float length;
	__host__ __device__ Vec3();
	__host__ __device__ Vec3(float _x, float _y, float _z);
	__host__ __device__ Vec3(const Vec3& v);

	__host__ __device__ ~Vec3();

	__host__ __device__ Vec3& operator=(const Vec3& v);
	__host__ __device__ Vec3& operator+=(const Vec3& v);
	__host__ __device__ Vec3& operator-=(const Vec3& v);
	__host__ __device__ Vec3& operator-();

	__host__ __device__ Vec3 operator+(const Vec3 & v);
	__host__ __device__ Vec3 operator-(const Vec3 & v);
	__host__ __device__ Vec3 operator*(const Vec3 & v);
	__host__ __device__ Vec3 operator/(const Vec3 & v);
	__host__ __device__ Vec3 operator*(const float & t);
	__host__ __device__ Vec3 operator/(const float & t);

	__host__ __device__ void normalize();

	__host__ __device__ static float dot(Vec3 v1, Vec3 v2);
	__host__ __device__ static Vec3 cross(Vec3 v1, Vec3 v2);

private:
	__host__ __device__ void updateLength();
};

class Normal
{
public:
	float x, y, z;

	__host__ __device__ Normal();
	__host__ __device__ Normal(float _x, float _y, float _z);
	__host__ __device__ Normal(const Normal& n);

	__host__ __device__ ~Normal();

	__host__ __device__ Normal& operator=(const Normal& n);
};

class Point2
{
public:
	float x, y;

	__host__ __device__ Point2();
	__host__ __device__ Point2(float _x, float _y);
	__host__ __device__ Point2(const Point2& p);
	__host__ __device__ ~Point2();

	__host__ __device__ Point2& operator= (const Point2& point);
};

class Point3
{
public:
	float x, y, z;

	__host__ __device__ Point3();
	__host__ __device__ Point3(float _x, float _y, float _z);
	__host__ __device__ Point3(const Point3& p);

	__host__ __device__ ~Point3();

	__host__ __device__ Point3& operator=(const Point3& p);
	__host__ __device__ Point3& operator+=(const Point3& p);
	__host__ __device__ Point3 operator/(const float& t);
};

struct BufferMap
{
public:
	BufferMap()
	{
		r = 0;
		g = 0;
		b = 0;
	}
	unsigned char b, g, r;
};

class Color
{
public:
	float r, g, b, a;

	__host__ __device__ Color();
	__host__ __device__ Color(float _r, float _g, float _b, float _a);
	__host__ __device__ Color(const Color& c);
	__host__ __device__ ~Color();

	__host__ __device__ Color& operator=(const Color& c);
	__host__ __device__ Color& operator+=(const Color& c);
	__host__ __device__ Color& operator-=(const Color& c);
	__host__ __device__ Color& operator*=(const Color& c);
	__host__ __device__ Color& operator*=(const float& t);
	__host__ __device__ Color& operator/=(const Color& c);
	__host__ __device__ Color& operator/=(const float& t);

	__host__ __device__ void toneMapping();
	__host__ __device__ void transToGamma();

	__host__ __device__ static Color lerp(Color c1, Color c2, float t);
	__host__ __device__ static Color GetBackgroundColor(float y)
	{
		float theta = acosf(y);
		float lerpFactor = theta / 3.14158f;
		return lerp(Color(0.32f, 0.64f, 1.0f, 1.0f), Color(0.65f, 0.65f, 0.65f, 1.0f), lerpFactor);
	}
};

class Record
{
public:
	float t;
	Color color;
	Normal normal;
	Point3 intersection;
	Point2 UV;
	int primitiveIndex;
	__host__ __device__ Record();
	__host__ __device__ Record(const Record& r);
	__host__ __device__ ~Record() {}

	__host__ __device__ Record& operator=(const Record& record);
};

class Ray
{
public:
	Point3 origin;
	Vec3 direction;
	Record record;
	__host__ __device__ Ray();
	__host__ __device__ Ray(const Point3& _origin, Vec3& _direction);
	__host__ __device__ Ray(const Ray& ray);
	__host__ __device__ ~Ray();


	__host__ __device__ Ray& operator=(const Ray& ray);
	__host__ __device__ void GetEndPoint();
};

class Quaternion
{
public:
	Vec3 imaginaryPart;
	float realPart;
	__host__ __device__ Quaternion();
	__host__ __device__ Quaternion(const float& _x, const float& _y, const float& _z, const float& _w);
	__host__ __device__ Quaternion(Vec3 axis, const float& theta);
	__host__ __device__ Quaternion(const Quaternion& q);

	__host__ __device__ Quaternion(const Vec3& v);
	__host__ __device__ Quaternion(const Point3& p);
	__host__ __device__ ~Quaternion();

	__host__ __device__ Quaternion operator*(const Quaternion & q);

	__host__ __device__ Quaternion conjugate();

	__host__ __device__ void rotatePoint(Point3& p);
	__host__ __device__ void rotateVector(Vec3& v);
};

class Texture
{
public:
	int width;
	int height;
	int nrChannel;
	unsigned char* data;
	__host__ __device__ Texture();
	__host__ __device__ ~Texture();

	__host__ __device__ Color get_UV_Color(float u, float v);
};
enum BRDF
{
	BlinPhong,
	DisneyPBR
};
class Material
{
public:
	BRDF brdf;

	__host__ __device__ Material& operator=(const Material& material);

#pragma region BlinPhong
	__host__  __device__ Material(BRDF _brdf, int _Albedo_index, int _Normal_index, int _Roughness_index, int _AO_index);
	__device__ Color InteractOnBlinPhone(Ray& ray);

#pragma endregion

#pragma region Disney PBR
	int Albedo_index;
	int Normal_index;
	int Metallic_index;
	int Roughness_index;
	int AO_index;
	__host__  __device__ Material(BRDF _brdf, int _Albedo_index, int _Normal_index, int _Metallic_index, int _Roughness_index, int _AO_index);
	__device__ Color InteractOnDisneyPBR(Ray& ray);
	__device__ Vec3 generateNextDirection(const float& specularContribution, const Vec3& viewDirection, const Vec3& normal);
	__host__ __device__ float DistributionGGX(const Vec3& N, const Vec3& H, float roughness);
	__host__ __device__ float GeometrySchlickGGX(float NdotV, float roughness);
	__host__ __device__ float GeometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness);
#pragma endregion

	__host__  __device__ Material();
	__host__  __device__ ~Material();

	__device__ Color Interact(Ray& ray);
	__host__ __device__ Vec3 ApplyNormalMap(const Normal& n, const Point2& uv);
};


enum Primitive_type
{
	Default,
	Sphere,
	Triangle,
	Plane
};
class Primitive
{
public:
#pragma region basic members
	Primitive_type type;
	Point3 centre;
	Color materialColor;
	Normal normal;
	Material material;
	__host__ __device__ Primitive();
	__host__ __device__ Primitive(
		Primitive_type _type,
		Point3 _centre,
		Color _materialColor,
		Normal _normal,
		float _radius,
		Point3 p0,
		Point3 p1,
		Point3 p2,
		Point2 uv0,
		Point2 uv1,
		Point2 uv2
	);
	__host__ __device__ ~Primitive();

	__host__ __device__ Primitive& operator=(const Primitive& primitive);

	__host__ __device__ bool HitTest(Ray& ray);
	__device__ Color getHitColor(Ray& ray);
#pragma endregion

#pragma region Sphere
	float radius;
	__host__ __device__ Primitive(Primitive_type Sphere, Point3 _centre, Color _materialColor, float _radius);
	__host__ __device__ bool HitTest_sphere(Ray& ray);
	__device__ Color getSphereHitColor(Ray& ray);
#pragma endregion

#pragma region Triangle
	Point3 points[3];
	Point2 uv[3];
	__host__ __device__ Primitive(Primitive_type Triangle, Color _materialColor, Point3 point0, Point3 point1, Point3 point2, Point2 uv0, Point2 uv1, Point2 uv2);
	__host__ __device__ bool HitTest_triangle(Ray& ray);
	__device__ Color getTriangleHitColor(Ray& ray);
	__host__ __device__ void updateNormal();
#pragma endregion

#pragma region Plane
	__host__ __device__ Primitive(Primitive_type Plane, Point3 _centre, Color _materialColor, Normal _normal);
	__host__ __device__ bool HitTest_plane(Ray& ray);
	__device__ Color getPlaneHitColor(Ray& ray);

#pragma endregion
};

struct primitive_linked_list
{
	Primitive primitive;
	primitive_linked_list* next_primitive_node;
};

enum Sampler
{
	regular
};
class Camera
{
public:
	Point3 position;
	float phi, theta;
	float fov, aspectRatio, viewDistance;
	Sampler sampler;

	__host__ __device__ Camera();
	__host__ __device__ Camera(Point3 _position, float _phi, float _theta, float _fov, float _aspectRatio, float _viewDistance, Sampler _sampler);
	__host__ __device__ Camera(Point3 _position, float rotation_x, float rotation_y, float rotation_z, float rotation_w, float _fov, float _aspectRatio, float _viewDistance, Sampler _sampler);
	__host__ __device__ Camera(const Camera& _camera);
	__host__ __device__ ~Camera() {}

	__host__ __device__ Camera& operator=(const Camera& camera);

	__host__ __device__ Ray getRay(float u, float v);
	__host__ __device__ float sample();

private:
	//int width, height;
	float viewPlaneWidth;
	float viewPlaneHeight;
	Quaternion rotation;
};


__global__ void change_device_primitive(int index, Primitive* device_primitive_list, Primitive changed)
{
	device_primitive_list[index] = changed;
}

class Scene
{
public:
	Primitive* device_primitive_list;//在device使用线性表，因为device基本上不需要增删，只负责渲染
	Camera camera;
	int primitives_num;
	__host__ __device__ Scene();
	__host__ __device__ Scene(Camera _camera, int _primitives_num);
	__host__ __device__ ~Scene() {}

	__host__ __device__ Scene& operator=(const Scene& scene);

	__host__ bool generate_device_scene();

	__host__ void change_device_scene(int index, Primitive changed);

	__host__ __device__ bool HitTest(Ray& ray);
	__device__ Color GetHitColor(Ray& ray);
};
#pragma endregion

#pragma region Vec2
Vec2::Vec2()
{
	x = 0.0f;
	y = 0.0f;
}

Vec2::Vec2(float _x, float _y)
{
	x = _x;
	y = _y;
}

Vec2::Vec2(const Vec2 & v)
{
	x = v.x;
	y = v.y;
}

Vec2::~Vec2()
{
}
#pragma endregion

#pragma region Vec3
Vec3::Vec3()
{
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
	length = 0.0f;
}

Vec3::Vec3(float _x, float _y, float _z)
{
	x = _x;
	y = _y;
	z = _z;
	updateLength();
}

Vec3::Vec3(const Vec3 & v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	length = v.length;
}

Vec3::~Vec3()
{
}

__host__ __device__ Vec3 & Vec3::operator=(const Vec3 & v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	length = v.length;
	return *this;
}

__host__ __device__ Vec3 & Vec3::operator+=(const Vec3 & v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	updateLength();
	return *this;
}

__host__ __device__ Vec3 & Vec3::operator-=(const Vec3 & v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	updateLength();
	return *this;
}

__host__ __device__ Vec3 & Vec3::operator-()
{
	x = -x;
	y = -y;
	z = -z;
	return *this;
}

__host__ __device__ Vec3 Vec3::operator+(const Vec3 & v)
{
	return Vec3(x + v.x, y + v.y, z + v.z);
}

__host__ __device__ Vec3 Vec3::operator-(const Vec3 & v)
{
	return Vec3(x - v.x, y - v.y, z - v.z);
}

__host__ __device__ Vec3 Vec3::operator*(const Vec3 & v)
{
	return Vec3(x * v.x, y * v.y, z * v.z);
}

__host__ __device__ Vec3 Vec3::operator/(const Vec3 & v)
{
	return Vec3(x / v.x, y / v.y, z / v.z);
}

__host__ __device__ Vec3 Vec3::operator*(const float & t)
{
	return Vec3(t * x, t *  y, t * z);
}

__host__ __device__ Vec3 Vec3::operator/(const float & t)
{
	return Vec3(x / t, y / t, z / t);
}

__host__ __device__ void Vec3::normalize()
{
	x /= length;
	y /= length;
	z /= length;
}

__host__ __device__ float Vec3::dot(Vec3 v1, Vec3 v2)
{
	return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

__host__ __device__ Vec3 Vec3::cross(Vec3 v1, Vec3 v2)
{
	return Vec3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

__host__ __device__ void Vec3::updateLength()
{
	length = sqrtf(x*x + y * y + z * z);
}

#pragma endregion

#pragma region Normal
Normal::Normal()
{
	x = 0.0f;
	y = 1.0f;
	z = 0.0f;
}

Normal::Normal(float _x, float _y, float _z)
{
	float length = sqrtf(_x*_x + _y * _y + _z * _z);
	x = _x / length;
	y = _y / length;
	z = _z / length;
}

Normal::Normal(const Normal & n)
{
	x = n.x;
	y = n.y;
	z = n.z;
}

Normal::~Normal()
{
}

__host__ __device__ Normal & Normal::operator=(const Normal & n)
{
	x = n.x;
	y = n.y;
	z = n.z;
	return *this;
}
#pragma endregion

#pragma region Point2
Point2::Point2()
{
	x = 0.0f;
	y = 0.0f;
}

Point2::Point2(float _x, float _y)
{
	x = _x;
	y = _y;
}

Point2::Point2(const Point2 & p)
{
	x = p.x;
	y = p.y;
}

Point2::~Point2()
{
}
Point2 & Point2::operator=(const Point2 & point)
{
	x = point.x;
	y = point.y;
	return *this;
}
#pragma endregion

#pragma region Point
Point3::Point3()
{
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
}

Point3::Point3(float _x, float _y, float _z)
{
	x = _x;
	y = _y;
	z = _z;
}

Point3::Point3(const Point3 & p)
{
	x = p.x;
	y = p.y;
	z = p.z;
}

Point3::~Point3()
{
}

__host__ __device__ Point3 & Point3::operator=(const Point3 & p)
{
	x = p.x;
	y = p.y;
	z = p.z;
	return *this;
}

__host__ __device__ Point3 & Point3::operator+=(const Point3 & p)
{
	x += p.x;
	y += p.y;
	z += p.z;
	return *this;
}

__host__ __device__ Point3 Point3::operator/(const float & t)
{
	return Point3(x / t, y / t, z / t);
}

#pragma endregion

#pragma region Color
Color::Color()
{
	r = 0.0f;
	g = 0.0f;
	b = 0.0f;
	a = 1.0f;
}

Color::Color(float _r, float _g, float _b, float _a)
{
	r = _r;
	g = _g;
	b = _b;
	a = _a;
}

Color::Color(const Color & c)
{
	r = c.r;
	g = c.g;
	b = c.b;
	a = 1.0f;
}

Color::~Color()
{
}

__host__ __device__ Color & Color::operator=(const Color & c)
{
	r = c.r;
	g = c.g;
	b = c.b;
	a = 1.0f;
	return *this;
}

__host__ __device__ Color & Color::operator+=(const Color & c)
{
	r += c.r;
	g += c.g;
	b += c.b;
	a = 1.0f;
	return *this;
}

__host__ __device__ Color & Color::operator-=(const Color & c)
{
	r -= c.r;
	g -= c.g;
	b -= c.b;
	a = 1.0f;
	return *this;
}

__host__ __device__ Color & Color::operator*=(const Color & c)
{
	r *= c.r;
	g *= c.g;
	b *= c.b;
	a = 1.0f;
	return *this;
}

__host__ __device__ Color & Color::operator*=(const float & t)
{
	r *= t;
	g *= t;
	b *= t;
	a = 1.0f;
	return *this;
}

__host__ __device__ Color & Color::operator/=(const Color & c)
{
	r /= c.r;
	g /= c.g;
	b /= c.b;
	a = 1.0f;
	return *this;
}

__host__ __device__ Color & Color::operator/=(const float & t)
{
	r /= t;
	g /= t;
	b /= t;
	a = 1.0f;
	return *this;
}

__host__ __device__ void Color::toneMapping()
{
	if (r > 1.0f)
	{
		r = 1.0f;
	}
	else if (g > 1.0f)
	{
		g = 1.0f;
	}
	else if (b > 1.0f)
	{
		b = 1.0f;
	}
}

__host__ __device__ void Color::transToGamma()
{
	r = sqrtf(r);
	g = sqrt(g);
	b = sqrt(b);
}

__host__ __device__ Color Color::lerp(Color c1, Color c2, float t)
{
	return t * (c2 - c1) + c1;
}
#pragma endregion

#pragma region Record
Record::Record()
{
	t = 0.0f;
	normal = Normal(0.0f, 1.0f, 0.0f);
	intersection = Point3(0.0f, 0.0f, 0.0f);
	UV = Point2(0.0f, 0.0f);
	color = Color(0.0f, 0.0f, 0.0f, 0.0f);
	primitiveIndex = -1;
}

Record::Record(const Record & r)
{
	t = r.t;
	normal = r.normal;
	intersection = r.intersection;
	UV = r.UV;
	color = r.color;
	primitiveIndex = r.primitiveIndex;
}

__host__ __device__ Record & Record::operator=(const Record& record)
{
	t = record.t;
	color = record.color;
	normal = record.normal;
	intersection = record.intersection;
	color = record.color;
	UV = record.UV;
	primitiveIndex = record.primitiveIndex;
	return *this;
}
#pragma endregion

#pragma region Ray
Ray::Ray()
{
	origin = Point3(0.0f, 0.0f, 0.0f);
	direction = Vec3(0.0f, 0.0f, -1.0f);
	record = Record();
}

Ray::Ray(const Point3 & _origin, Vec3 & _direction)
{
	origin = _origin;
	_direction.normalize();
	direction = _direction;
}

Ray::Ray(const Ray & ray)
{
	origin = ray.origin;
	direction = ray.direction;
	record = ray.record;
}

Ray::~Ray()
{
}

__host__ __device__ Ray & Ray::operator=(const Ray& ray)
{
	origin = ray.origin;
	direction = ray.direction;
	record = ray.record;
	return *this;
}

__host__ __device__ void Ray::GetEndPoint()
{
	record.intersection = Point3(origin + record.t * direction);
}
#pragma endregion

#pragma region Quaternion
Quaternion::Quaternion()
{
	imaginaryPart = Vec3();
	realPart = 1.0f;
}

Quaternion::Quaternion(const float & _x, const float & _y, const float & _z, const float & _w)
{
	imaginaryPart.x = _x;
	imaginaryPart.y = _y;
	imaginaryPart.z = _z;
	realPart = _w;
}

Quaternion::Quaternion(Vec3 axis, const float& theta)
{
	axis.normalize();
	imaginaryPart = axis * sinf(0.5f * theta);
	realPart = cosf(0.5f * theta);
}

Quaternion::Quaternion(const Quaternion & q)
{
	imaginaryPart = q.imaginaryPart;
	realPart = q.realPart;
}

Quaternion::Quaternion(const Vec3 & v)
{
	imaginaryPart.x = v.x;
	imaginaryPart.y = v.y;
	imaginaryPart.z = v.z;
	realPart = 0.0f;
}

Quaternion::Quaternion(const Point3 & p)
{
	imaginaryPart.x = p.x;
	imaginaryPart.y = p.y;
	imaginaryPart.z = p.z;
	realPart = 0.0f;
}

Quaternion::~Quaternion()
{
}

__host__ __device__ Quaternion Quaternion::operator*(const Quaternion & q)
{
	Vec3 result_imaginary_part = realPart * q.imaginaryPart + q.realPart*imaginaryPart + Vec3::cross(imaginaryPart, q.imaginaryPart);
	float result_real_part = realPart * q.realPart - Vec3::dot(imaginaryPart, q.imaginaryPart);
	return Quaternion(result_imaginary_part.x, result_imaginary_part.y, result_imaginary_part.z, result_real_part);
}

__host__ __device__ Quaternion Quaternion::conjugate()
{
	return Quaternion(-imaginaryPart.x, -imaginaryPart.y, -imaginaryPart.z, realPart);
}

__host__ __device__ void Quaternion::rotatePoint(Point3 & p)
{
	Quaternion point(p);
	point = *this * point * this->conjugate();
	p.x = point.imaginaryPart.x;
	p.y = point.imaginaryPart.y;
	p.z = point.imaginaryPart.z;
}

__host__ __device__ void Quaternion::rotateVector(Vec3 & v)
{
	Quaternion vector(v);
	vector = *this * vector * this->conjugate();
	v.x = vector.imaginaryPart.x;
	v.y = vector.imaginaryPart.y;
	v.z = vector.imaginaryPart.z;
}

#pragma endregion

#pragma region Texture
Texture::Texture()
{
}

Texture::~Texture()
{
}

Color Texture::get_UV_Color(float u, float v)
{

	int index = (int)(4 * (roundf(v*(height-1))*width + roundf(u*(width-1))));

	float r = (float)data[index] / 255.0f;
	float g = (float)data[index + 1] / 255.0f;
	float b = (float)data[index + 2] / 255.0f;
	//float a = (float)data[index + 3] / 255.0f;
	float a = 1.0f;
	return Color(r, g, b, a);
}

#pragma endregion

#pragma region Material
__host__ __device__ Material & Material::operator=(const Material & material)
{
	brdf = material.brdf;
	Albedo_index = material.Albedo_index;
	Normal_index = material.Normal_index;
	Metallic_index = material.Metallic_index;
	Roughness_index = material.Roughness_index;
	AO_index = material.AO_index;
	return *this;
}
Material::Material(BRDF _brdf, int _Albedo_index, int _Normal_index, int _Roughness_index, int _AO_index)
{
	brdf = _brdf;
	Albedo_index = _Albedo_index;
	Normal_index = _Normal_index;
	Roughness_index = _Roughness_index;
	AO_index = _AO_index;
}
__device__ Color Material::InteractOnBlinPhone(Ray & ray)
{
	Point2 uv = ray.record.UV;

	Color Albedo = device_texture_list[Albedo_index].get_UV_Color(uv.x, uv.y);
	Vec3 normal = ApplyNormalMap(ray.record.normal, uv);
	float metalness = device_texture_list[Metallic_index].get_UV_Color(uv.x, uv.y).r;
	float roughness = device_texture_list[Roughness_index].get_UV_Color(uv.x, uv.y).r;
	float AO = device_texture_list[AO_index].get_UV_Color(uv.x, uv.y).r;

	Vec3 viewDirection = -ray.direction;
	Vec3 lightDirection = 2.0f * Vec3::dot(normal, viewDirection)*normal - viewDirection;
	Vec3 halfDirection = 0.5f*(lightDirection + viewDirection);
	float costheta = Vec3::dot(viewDirection, normal);

	Color diffuse = Albedo * AO;

	float surfaceColorIntensity = (Albedo.r + Albedo.g + Albedo.b) / 3.0f;
	float F0 = mix(0.04f, surfaceColorIntensity, metalness);
	float F = F0 + (1.0f - F0)* powf(1.0f - costheta, 5.0f);

	float specularContribution = F;

	ray.direction = generateNextDirection(specularContribution, viewDirection, normal);

	return diffuse;
}
Material::Material(BRDF _brdf, int _Albedo_index, int _Normal_index, int _Metallic_index, int _Roughness_index, int _AO_index)
{
	brdf = _brdf;
	Albedo_index = _Albedo_index;
	Normal_index = _Normal_index;
	Metallic_index = _Metallic_index;
	Roughness_index = _Roughness_index;
	AO_index = _AO_index;
}
__device__ Color Material::InteractOnDisneyPBR(Ray & ray)
{
	Point2 uv = ray.record.UV;

	Color Albedo = device_texture_list[Albedo_index].get_UV_Color(uv.x, uv.y);
	Vec3 normal = ApplyNormalMap(ray.record.normal, uv);
	float metalness = device_texture_list[Metallic_index].get_UV_Color(uv.x, uv.y).r;
	float roughness = device_texture_list[Roughness_index].get_UV_Color(uv.x, uv.y).r;
	float AO = device_texture_list[AO_index].get_UV_Color(uv.x, uv.y).r;

	Vec3 viewDirection = -ray.direction;
	Vec3 lightDirection = 2.0f * Vec3::dot(normal, viewDirection)*normal - viewDirection;
	Vec3 halfDirection = 0.5f*(lightDirection + viewDirection);
	float costheta = Vec3::dot(viewDirection, normal);

	Color diffuse = Albedo * AO;

	//因为暂时无法单独的对某个波长进行追踪，所以这个先弃用
	//Color F0 = mix(Color(0.04f, 0.04f, 0.04f, 1.0f), surfaceColor, metalness);
	//Color fresnelSchlick = F0 + (1.0f - F0)* powf(1.0f - costheta, 5.0f);
	float surfaceColorIntensity = (Albedo.r + Albedo.g + Albedo.b) / 3.0f;
	float F0 = mix(0.04f, surfaceColorIntensity, metalness);
	float F = F0 + (1.0f - F0)* powf(1.0f - costheta, 5.0f);

	float NDF = DistributionGGX(normal, halfDirection, roughness);
	float G = GeometrySmith(normal, viewDirection, lightDirection, roughness);

	float nominator = NDF * G * F;
	float denominator = 4.0 * fmaxf(Vec3::dot(normal, viewDirection), 0.0) * fmaxf(Vec3::dot(normal, lightDirection), 0.0f) + 0.001f;
	float specularContribution = nominator / denominator;

	ray.direction = generateNextDirection(specularContribution, viewDirection, normal);
	return diffuse;

#pragma region unknown
	//float kS = F;
	//float kD = 1.0f - kS;

	//kD *= 1.0f - metalness;

	//float NdotL = fmaxf(Vec3::dot(normal, nextDirecton), 0.0);
	//float Lo = (kD * albedo / PI + specular) * radiance * NdotL;
#pragma endregion
}
__device__ Vec3 Material::generateNextDirection(const float& specularContribution, const Vec3& viewDirection, const Vec3& normal)
{
	int globalIdx = blockIdx.x*blockDim.x + threadIdx.x;

	static int i = 0;
	if (i >= 4)
	{
		i = 0;
	}

	int currentRandNumIndex = (depth* globalIdx + i) % device__randNumSize;

	float currentRandNumber = device_randNumber_ptr[currentRandNumIndex];
	Vec3 result;
	if (specularContribution > currentRandNumber)
	{
		result = 2.0f*Vec3::dot(viewDirection, normal) * normal - viewDirection;
	}
	else
	{
		int currentRandHemisphereVectorIndex = (depth* globalIdx + i) % device_randHemisphereVectorSize;
		result = device_randHemisphereVector_ptr[currentRandHemisphereVectorIndex];
	}

	return result;
}
__host__ __device__ float Material::DistributionGGX(const Vec3 & N, const Vec3 & H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = fmaxf(Vec3::dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / denom;
}
__host__ __device__ float Material::GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}
__host__ __device__ float Material::GeometrySmith(const Vec3 & N, const Vec3 & V, const Vec3 & L, float roughness)
{
	float NdotV = fmaxf(Vec3::dot(N, V), 0.0);
	float NdotL = fmaxf(Vec3::dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}
Material::Material()
{
	Albedo_index = -1;
	Normal_index = -1;
	Metallic_index = -1;
	Roughness_index = -1;
	AO_index = -1;
}

Material::~Material()
{
}

__device__ Color Material::Interact(Ray& ray)
{
	Color result;
	switch (brdf)
	{
	case BRDF::BlinPhong: result = InteractOnBlinPhone(ray); break;
	case BRDF::DisneyPBR: result = InteractOnDisneyPBR(ray); break;
	}
	return result;
}

__host__ __device__ Vec3 Material::ApplyNormalMap(const Normal & n, const Point2 & uv)
{
	Vec3 normal;
	normal.x = n.x;
	normal.y = n.y;
	normal.z = n.z;
	Vec3 axis = Vec3::cross(Vec3(0.0f, 1.0f, 0.0f), normal);
	float theta = fabsf(Vec3::dot(normal, Vec3(0.0f, 1.0f, 0.0f)));
	Quaternion rotation(axis, theta);
	Color normal_T_temp = device_texture_list[Normal_index].get_UV_Color(uv.x, uv.y);
	Vec3 normal_T;
	normal_T.x = normal_T_temp.r;
	normal_T.y = normal_T_temp.g;
	normal_T.z = normal_T_temp.b;

	rotation.rotateVector(normal_T);

	return normal_T;
}

#pragma endregion

#pragma region Primitive
Primitive::Primitive()
{
	type = Primitive_type::Default;
	centre = Point3();
	materialColor = Color();
}

Primitive::Primitive(Primitive_type _type, Point3 _centre, Color _materialColor, Normal _normal, float _radius, Point3 p0, Point3 p1, Point3 p2, Point2 uv0, Point2 uv1, Point2 uv2)
{
	type = _type;
	centre = _centre;
	materialColor = _materialColor;
	normal = _normal;
	radius = _radius;
	points[0] = p0;
	points[1] = p1;
	points[2] = p2;
	uv[0] = uv0;
	uv[1] = uv1;
	uv[2] = uv2;
}

Primitive::~Primitive()
{
}

__host__ __device__ Primitive & Primitive::operator=(const Primitive & primitive)
{
	type = primitive.type;
	centre = primitive.centre;
	materialColor = primitive.materialColor;
	material = primitive.material;
	normal = primitive.normal;
	radius = primitive.radius;
	points[0] = primitive.points[0];
	points[1] = primitive.points[1];
	points[2] = primitive.points[2];
	uv[0] = primitive.uv[0];
	uv[1] = primitive.uv[1];
	uv[2] = primitive.uv[2];
	return *this;
}

__host__ __device__ bool Primitive::HitTest(Ray & ray)
{
	bool isIntersected = false;

	switch (type)
	{
	case Default:break;
	case Sphere:isIntersected = HitTest_sphere(ray); break;
	case Triangle:isIntersected = HitTest_triangle(ray); break;
	case Plane:isIntersected = HitTest_plane(ray); break;
	}
	return isIntersected;
}

__device__ Color Primitive::getHitColor(Ray& ray)
{
	Color result;
	switch (type)
	{
	case Sphere:result = getSphereHitColor(ray); break;
	case Triangle:result = getTriangleHitColor(ray); break;
	case Plane:result = getPlaneHitColor(ray); break;
	}
	return result;
}

Primitive::Primitive(Primitive_type Sphere, Point3 _centre, Color _materialColor, float _radius)
{
	type = Primitive_type::Sphere;

	centre = _centre;
	materialColor = _materialColor;
	radius = _radius;
}

__host__ __device__ bool Primitive::HitTest_sphere(Ray & ray)
{
	ray.direction.normalize();
	Vec3 CO = ray.origin - centre;
	float A = Vec3::dot(ray.direction, ray.direction);
	float B = 2 * Vec3::dot(ray.direction, CO);
	float C = Vec3::dot(CO, CO) - radius * radius;
	float discriminant = B * B - 4 * A * C;

	if (discriminant > 0.0f)
	{
		float result = (-B - sqrtf(discriminant)) / (2 * A);
		if (result < FLT_MAX && result > 0.001f)
		{
			ray.record.t = result;
			ray.GetEndPoint();
			Vec3 n = (ray.record.intersection - centre) / radius;
			ray.record.normal.x = n.x;
			ray.record.normal.y = n.y;
			ray.record.normal.z = n.z;

			float u = acosf(n.z / sqrtf(n.x*n.x + n.z*n.z));
			float v = (asinf(n.y) + 0.5f * PI) / PI;
			//printf("Sphere uv: %f, %f.\n", u, v);

			ray.record.UV = Point2(u, v);

			return true;
		}
		result = (-B + sqrtf(discriminant)) / (2 * A);
		if (result < FLT_MAX && result > 0.001f)
		{
			ray.record.t = result;
			ray.GetEndPoint();
			Vec3 n = (ray.record.intersection - centre) / radius;
			ray.record.normal.x = n.x;
			ray.record.normal.y = n.y;
			ray.record.normal.z = n.z;

			float u = acosf(n.z / sqrtf(n.x*n.x + n.z*n.z));
			float v = (asinf(n.y) + 0.5f * PI) / PI;

			//printf("Sphere uv: %f, %f.\n", u, v);
			ray.record.UV = Point2(u, v);

			return true;
		}
	}
	return false;
}

__device__ Color Primitive::getSphereHitColor(Ray& ray)
{
	Color result = material.Interact(ray);
	return materialColor * result;
}

Primitive::Primitive(Primitive_type Triangle, Color _materialColor, Point3 point0, Point3 point1, Point3 point2, Point2 uv0, Point2 uv1, Point2 uv2)
{
	type = Primitive_type::Triangle;

	centre = (point0 + point1 + point2) / 3.0f;
	materialColor = _materialColor;
	points[0] = point0;
	points[1] = point1;
	points[2] = point2;
	uv[0] = uv0;
	uv[1] = uv1;
	uv[2] = uv2;
	Vec3 a = points[1] - points[0];
	Vec3 b = points[2] - points[0];
	Vec3 n = Vec3::cross(a, b);
	n.normalize();
	normal.x = n.x;
	normal.y = n.y;
	normal.z = n.z;
}

__host__ __device__ bool Primitive::HitTest_triangle(Ray & ray)
{
	float a = points[0].x - points[1].x, b = points[0].x - points[2].x, c = ray.direction.x, d = points[0].x - ray.origin.x;
	float e = points[0].y - points[1].y, f = points[0].y - points[2].y, g = ray.direction.y, h = points[0].y - ray.origin.y;
	float i = points[0].z - points[1].z, j = points[0].z - points[2].z, k = ray.direction.z, l = points[0].z - ray.origin.z;

	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0 / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0.001f)
		return (false);

	float r = r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0.001f)
		return (false);

	if (beta + gamma > 1.0f)
		return (false);

	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if (t < 0.001f)
		return false;

	ray.record.t = t;
	ray.record.normal = normal;
	ray.record.intersection = ray.origin + t * ray.direction;
	float px = ray.record.intersection.x;
	float py = ray.record.intersection.y;
	float pz = ray.record.intersection.z;
	float adjoint = 1.0f / (uv[0].x*uv[1].y - uv[1].x*uv[0].y);
	float u = adjoint * (px*(points[0].x*uv[1].y - points[1].x*uv[0].y) + py * (points[0].y*uv[1].y - points[1].y*uv[0].y) + pz * (points[0].z*uv[1].y - points[1].z*uv[0].y));
	float v = -adjoint * (px*(points[0].x*uv[1].x - points[1].x*uv[0].x) + py * (points[0].y*uv[1].x - points[1].y*uv[0].x) + pz * (points[0].z*uv[1].x - points[1].z*uv[0].x));

	ray.record.UV = Point2(u, v);
	return true;
}

__device__ Color Primitive::getTriangleHitColor(Ray& ray)
{
	Color result = material.Interact(ray);
	return materialColor * result;

}

__host__ __device__ void Primitive::updateNormal()
{
	Vec3 v1 = points[1] - points[0];
	Vec3 v2 = points[2] - points[0];
	Vec3 n = Vec3::cross(v1, v2);
	normal.x = n.x;
	normal.y = n.y;
	normal.z = n.z;
}

Primitive::Primitive(Primitive_type Plane, Point3 _centre, Color _materialColor, Normal _normal)
{
	type = Primitive_type::Plane;
	centre = _centre;
	materialColor = _materialColor;
	normal = _normal;
}

__host__ __device__ bool Primitive::HitTest_plane(Ray & ray)
{
	//float t = (centre - ray.origin) * planeNormal / (ray.direction * planeNormal);
	Vec3 o_to_c = centre - ray.origin;
	Vec3 n;
	n.x = normal.x;
	n.y = normal.y;
	n.z = normal.z;
	float t = Vec3::dot(o_to_c, n) / Vec3::dot(ray.direction, n);
	if (t > 0.001f) {
		ray.record.t = t;
		ray.record.normal = normal;
		ray.record.intersection = ray.origin + t * ray.direction;

		float u = fmodf(fabsf(ray.record.intersection.x), 1.0f);
		float v = fmodf(fabsf(ray.record.intersection.z), 1.0f);

		ray.record.UV = Point2(u, v);

		return true;
	}
	return false;
}

__device__ Color Primitive::getPlaneHitColor(Ray& ray)
{
	Color result = material.Interact(ray);
	return materialColor * result;

#pragma region Tess
	//Color result;

	//float x = record.intersection.x;
	//float z = record.intersection.z;

	//if (x<0.0f)
	//{
	//	x = x - 1.0f;
	//}
	//if (z<0.0f)
	//{
	//	z = z - 1.0f;
	//}
	//x = fabsf(x);
	//z = fabsf(z);

	//x = fmodf(x, 2.0f);
	//z = fmodf(z, 2.0f);
	//bool x_test = fmodf(x, 2.0f) < 1.0f;
	//bool z_test = fmodf(z, 2.0f) < 1.0f;
	//bool final = !(x_test ^ z_test);
	//switch (final)
	//{
	//case true:result = Color(1.0f, 1.0f, 1.0f, 1.0f); break;
	//case false:result = Color(0.0f, 0.0f, 0.0f, 1.0f); break;
	//}
	//return result;
#pragma endregion
}
#pragma endregion

#pragma region Camera
Camera::Camera()
{
	position = Point3();
	phi = 0.0f;
	theta = 0.0f;
	//width = 1280;
	//height = 720;
	fov = 0.5*3.1415926f;
	aspectRatio = 16.0f / 9.0f;
	viewDistance = 5.0f;
	sampler = Sampler::regular;
}

Camera::Camera(Point3 _position, float _phi, float _theta, float _fov, float _aspectRatio, float _viewDistance, Sampler _sampler)
{
	position = _position;
	phi = _phi;
	theta = _theta;
	//width = _width;
	//height = _height;
	fov = _fov;
	aspectRatio = _aspectRatio;
	viewDistance = _viewDistance;
	sampler = _sampler;

	viewPlaneWidth = 2.0f*tanf(0.5f*_fov)*_viewDistance;
	viewPlaneHeight = viewPlaneWidth / _aspectRatio;
}

Camera::Camera(Point3 _position, float rotation_x, float rotation_y, float rotation_z, float rotation_w, float _fov, float _aspectRatio, float _viewDistance, Sampler _sampler)
{
	position = _position;
	rotation.imaginaryPart.x = rotation_x;
	rotation.imaginaryPart.y = rotation_y;
	rotation.imaginaryPart.z = rotation_z;
	rotation.realPart = rotation_w;

	fov = _fov;
	aspectRatio = _aspectRatio;
	viewDistance = _viewDistance;
	sampler = _sampler;

	viewPlaneWidth = 2.0f*tanf(0.5f*_fov)*_viewDistance;
	viewPlaneHeight = viewPlaneWidth / _aspectRatio;
}

Camera::Camera(const Camera & camera)
{
	position = camera.position;
	phi = camera.phi;
	theta = camera.theta;
	//width = camera.width;
	//height = camera.height;
	fov = camera.fov;
	aspectRatio = camera.aspectRatio;
	viewDistance = camera.viewDistance;
	sampler = camera.sampler;
	viewPlaneWidth = camera.viewPlaneWidth;
	viewPlaneHeight = camera.viewPlaneHeight;
	rotation = camera.rotation;
}

__host__ __device__ Camera & Camera::operator=(const Camera & camera)
{
	position = camera.position;
	phi = camera.phi;
	theta = camera.theta;
	//width = camera.width;
	//height = camera.height;
	fov = camera.fov;
	aspectRatio = camera.aspectRatio;
	viewDistance = camera.viewDistance;
	sampler = camera.sampler;
	viewPlaneWidth = camera.viewPlaneWidth;
	viewPlaneHeight = camera.viewPlaneHeight;
	rotation = camera.rotation;
	return *this;
}

__host__ __device__ Ray Camera::getRay(float u, float v)
{
	//View坐标系下的ViewPlanePos
	Point3 viewPlanePos_V(-viewPlaneWidth * (u - 0.5f + 0.5f * 1.0f / width), viewDistance, -viewPlaneHeight * (v - 0.5f + 0.5f * 1.0f / height));
	rotation.rotatePoint(viewPlanePos_V);
	Point3 viewPlanePos_W = viewPlanePos_V;
	viewPlanePos_W += position;
	Vec3 direction = viewPlanePos_W - position;
	return Ray(position, direction);
}

__host__ __device__ float Camera::sample()
{
	return 0.5f;
}
#pragma endregion

#pragma region Scene
Scene::Scene()
{
	device_primitive_list = 0;
	camera = Camera();
	primitives_num = 0;
}

Scene::Scene(Camera _camera, int _primitives_num)
{
	camera = _camera;
	primitives_num = _primitives_num;
}

__host__ __device__ Scene & Scene::operator=(const Scene & scene)
{
	device_primitive_list = scene.device_primitive_list;
	camera = scene.camera;
	primitives_num = scene.primitives_num;
	return *this;
}

__global__ void checkprimitivetype(Primitive_type* device_type, Primitive* list, int index)
{
	*device_type = list[index].type;
}

__global__ void teee(float* device_points, Primitive* device_primitive_list)
{
	device_points[0] = device_primitive_list[0].points[0].x;
	device_points[1] = device_primitive_list[0].points[0].y;
	device_points[2] = device_primitive_list[0].points[0].z;
	device_points[3] = device_primitive_list[0].points[1].x;
	device_points[4] = device_primitive_list[0].points[1].y;
	device_points[5] = device_primitive_list[0].points[1].z;
	device_points[6] = device_primitive_list[0].points[2].x;
	device_points[7] = device_primitive_list[0].points[2].y;
	device_points[8] = device_primitive_list[0].points[2].z;

}
__host__ bool Scene::generate_device_scene()
{
	bool isGenerateSceneSuccess = true;

	Primitive* host_primitive_list = new Primitive[primitives_num];

	cudaMalloc(&device_primitive_list, primitives_num * sizeof(Primitive));

	for (int i = 0; i < primitives_num; i++)
	{
		host_primitive_list[i] = primitives[i];
		//std::cout << "当前输入的primitive.uv[0]: " << host_primitive_list[i].uv[0].x << "\t" << host_primitive_list[i].uv[0].y << std::endl;
		//std::cout << "当前输入的primitive.uv[1]: " << host_primitive_list[i].uv[1].x << "\t" << host_primitive_list[i].uv[1].y << std::endl;
		//std::cout << "当前输入的primitive.uv[2]: " << host_primitive_list[i].uv[2].x << "\t" << host_primitive_list[i].uv[2].y << std::endl;
	}
	cudaError_t error = cudaMemcpy(device_primitive_list, host_primitive_list, primitives_num * sizeof(Primitive), cudaMemcpyHostToDevice);
	if (error != cudaError_t::cudaSuccess)
	{
		isGenerateSceneSuccess = false;
		printf("CUDA Error: Host To Device Cpy出错！%s\n", cudaGetErrorString(error));
	}

	float* points = new float[9];
	float* device_uvs;
	cudaMalloc(&device_uvs, 9 * sizeof(float));
	//teee << <1, 1 >> > (device_uvs, device_primitive_list);
	//cudaMemcpy(points, device_uvs, 9 * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 9; i++)
	//{
	//	std::cout << points[i] << std::endl;
	//}
	//delete[] points;
	//cudaFree(device_uvs);

	delete[] host_primitive_list;
	return isGenerateSceneSuccess;
}

__host__ void Scene::change_device_scene(int index, Primitive changed)
{
	change_device_primitive << <1, 1 >> > (index, device_primitive_list, changed);
}

__host__ __device__ bool Scene::HitTest(Ray & ray)
{
	Ray original = ray;
	Ray localRay = original;
	Ray resultRay = original;
	float minDistance = FLT_MAX;
	bool haveHit = false;
	for (int i = 0; i < primitives_num; i++)
	{
		localRay = original;
		if (device_primitive_list[i].HitTest(localRay))
		{
			if (localRay.record.t < minDistance)
			{
				haveHit = true;
				minDistance = localRay.record.t;
				localRay.record.primitiveIndex = i;
				resultRay = localRay;
			}
		}
	}
	if (haveHit)
	{
		ray = resultRay;
	}
	return haveHit;
}

__device__ Color Scene::GetHitColor(Ray& ray)
{
	return device_primitive_list[ray.record.primitiveIndex].getHitColor(ray);
}
#pragma endregion

#pragma region Common Alogrithm
inline __host__ __device__ Vec3 operator*(const float & t, const Vec3 & v)
{
	return Vec3(t * v.x, t *  v.y, t * v.z);
}

inline __host__ __device__ Vec3 operator-(const Point3 & p1, const Point3 & p2)
{
	return Vec3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

inline __host__ __device__ Point3 operator+(const Point3 & p, const Vec3 & v)
{
	return Point3(p.x + v.x, p.y + v.y, p.z + v.z);
}

inline __host__ __device__ Point3 operator+(const Point3 & p1, const Point3 & p2)
{
	return Point3(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
}

inline __host__ __device__ Point3 operator-(const Point3 & p, const Vec3 & v)
{
	return Point3(p.x - v.x, p.y - v.y, p.z - v.z);
}

inline __host__ __device__ Color operator+(const Color & c1, const Color & c2)
{
	return Color(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b, 1.0f);
}

inline __host__ __device__ Color operator+(const Color & c, const float & t)
{
	return Color(c.r + t, c.g + t, c.b + t, 1.0f);
}

inline __host__ __device__ Color operator+(const float & t, const Color & c)
{
	return Color(c.r + t, c.g + t, c.b + t, 1.0f);
}

inline __host__ __device__ Color operator-(const Color & c1, const Color & c2)
{
	return Color(c1.r - c2.r, c1.g - c2.g, c1.b - c2.b, 1.0f);
}

inline __host__ __device__ Color operator-(const Color & c, const float & t)
{
	return Color(c.r - t, c.g - t, c.b - t, 1.0f);
}

inline __host__ __device__ Color operator-(const float & t, const Color & c)
{
	return Color(t - c.r, t - c.g, t - c.b, 1.0f);
}

inline __host__ __device__ Color operator*(const Color & c1, const Color & c2)
{
	return Color(c1.r * c2.r, c1.g * c2.g, c1.b * c2.b, 1.0f);
}

inline __host__ __device__ Color operator*(const float & t, const Color & c)
{
	return Color(t * c.r, t * c.g, t * c.b, 1.0f);
}

inline __host__ __device__ Color operator*(const Color & c, const float & t)
{
	return Color(t * c.r, t * c.g, t * c.b, 1.0f);
}

inline __host__ __device__ Color operator/(const Color & c1, const Color & c2)
{
	return Color(c1.r / c2.r, c1.g / c2.g, c1.b / c2.b, 1.0f);
}

inline __host__ __device__ Color operator/(const Color & c, const float & t)
{
	return Color(c.r / t, c.g / t, c.b / t, 1.0f);
}

inline __host__ __device__ Color mix(const Color & c1, const Color & c2, float t)
{
	return c1 * (1.0f - t) + t * c2;
}

inline __host__ __device__ float mix(const float & t1, const float & t2, float t)
{
	return t1 * (1.0f - t) + t * t2;
}

#pragma endregion
#endif // !__CUDAUTIL__CUH