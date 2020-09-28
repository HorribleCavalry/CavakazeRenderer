#ifndef __CUDA3DMATH__CUH__
#define __CUDA3DMATH__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "../CudaSTD/cuvector.cuh"
#include "../CudaSTD/cuiostream.cuh"
#include <float.h>
#include <math.h>

#define PI 3.14159265358979323846
#define Epsilon 0.0009765625

namespace CUM
{
#define LogData(data) logData(data)

#pragma region Vec2
	template<typename T>
	class Vec2
	{
	public:
		T x, y;
	public:
		__duel__ Vec2() :x(0), y(0) {}
		__duel__ Vec2(const T& n) :x(n), y(n) {}
		__duel__ Vec2(const T& _x, const T& _y) :x(_x), y(_y) {}
		__duel__ Vec2(const Vec2<T>& v) : x(v.x), y(v.y) {}
		template<typename U>
		__duel__ explicit Vec2(const Vec2<U>& v) : x(v.x), y(v.y) {}
		__duel__ ~Vec2() {}
	public:
		__duel__ const Vec2& operator=(const Vec2<int>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 1, "The <idx> in Vec2<T>::operator[idx] is illegal!");
			return idx == 0 ? x : y;
		}
	};

	typedef Vec2<Int> Vec2i;
	typedef Vec2<Float> Vec2f;

#pragma region Vec2 vector operation

	template<typename T>
	__duel__ const Vec2<Int> floor(const Vec2<T>& v)
	{
		return Vec2<Int>(floor(v.x), floor(v.y));
	}

	template<typename T>
	__duel__ const T dot(const Vec2<T>& v0, const Vec2<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y;
	}

	template<typename T, typename U>
	__duel__ Float dot(const Vec2<T>& v0, const Vec2<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y;
	}

	template<typename T>
	__duel__ const Vec2<Float> normalize(const Vec2<T>& vec)
	{
		Float square = vec.x*vec.x + vec.y*vec.y;
		CHECK(square > 0.0, "Vec2 normalize error: square can not less than 0.0!");
		Float norm = sqrt(square);
		Float inv = 1.0 / norm;
		return inv * vec;
	}
#pragma endregion

#pragma region Vec2 same type operation

#pragma region Vec2 same type operation +

	template<typename T>
	__duel__ const Vec2<T> operator+(const T& n, const Vec2<T>& v)
	{
		return Vec2<T>(n + v.x, n + v.y);
	}
	template<typename T>
	__duel__ const Vec2<T> operator+(const Vec2<T>& v, const T& n)
	{
		return Vec2<T>(v.x + n, v.y + n);
	}
	template<typename T>
	__duel__ const Vec2<T> operator+(const Vec2<T>& v0, const Vec2<T>& v1)
	{
		return Vec2<T>(v0.x + v1.x, v0.y + v1.y);
	}

	template<typename T, typename U>
	__duel__ const Vec2<T>& operator+=(Vec2<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec2<T>& operator+=(Vec2<T>& v0, const Vec2<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		return v0;
	}

	__duel__ const Vec2<Int>& operator+=(Vec2<Int>& v, const Float& n) = delete;
	__duel__ const Vec2<Int>& operator+=(Vec2<Int>& v0, const Vec2<Float>& v1) = delete;


#pragma endregion

#pragma region Vec2 same type operation -

	template<typename T>
	__duel__ const Vec2<T> operator-(const T& n, const Vec2<T>& v)
	{
		return Vec2<T>(n - v.x, n - v.y);
	}
	template<typename T>
	__duel__ const Vec2<T> operator-(const Vec2<T>& v, const T& n)
	{
		return Vec2<T>(v.x - n, v.y - n);
	}
	template<typename T>
	__duel__ const Vec2<T> operator-(const Vec2<T>& v0, const Vec2<T>& v1)
	{
		return Vec2<T>(v0.x - v1.x, v0.y - v1.y);
	}

	template<typename T>
	__duel__ const Vec2<T>& operator-=(Vec2<T>& v, const T& n)
	{
		v.x -= n;
		v.y -= n;
		return v;
	}
	template<typename T>
	__duel__ const Vec2<T>& operator-=(Vec2<T>& v0, const Vec2<T>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		return v0;
	}

	__duel__ const Vec2<Int>& operator-=(Vec2<Int>& v, const Float& n) = delete;
	__duel__ const Vec2<Int>& operator-=(Vec2<Int>& v0, const Vec2<Float>& v1) = delete;

#pragma endregion

#pragma region Vec2 same type operation *

	template<typename T>
	__duel__ const Vec2<T> operator*(const T& n, const Vec2<T>& v)
	{
		return Vec2<T>(n * v.x, n * v.y);
	}
	template<typename T>
	__duel__ const Vec2<T> operator*(const Vec2<T>& v, const T& n)
	{
		return Vec2<T>(v.x * n, v.y * n);
	}
	template<typename T>
	__duel__ const Vec2<T> operator*(const Vec2<T>& v0, const Vec2<T>& v1)
	{
		return Vec2<T>(v0.x * v1.x, v0.y * v1.y);
	}

	template<typename T>
	__duel__ const Vec2<T>& operator*=(Vec2<T>& v, const T& n)
	{
		v.x *= n;
		v.y *= n;
		return v;
	}
	template<typename T>
	__duel__ const Vec2<T>& operator*=(Vec2<T>& v0, const Vec2<T>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		return v0;
	}

	__duel__ const Vec2<Int>& operator*=(Vec2<Int>& v, const Float& n) = delete;
	__duel__ const Vec2<Int>& operator*=(Vec2<Int>& v0, const Vec2<Float>& v1) = delete;

#pragma endregion

#pragma region Vec2 same type operation /

	template<typename T>
	__duel__ const Vec2<T> operator/(const T& n, const Vec2<T>& v)
	{
		CHECK(v.x != 0, "Same type Vec2 operator/ error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type Vec2 operator/ error: v.y can not be 0!");
		return Vec2<T>(n / v.x, n / v.y);
	}
	template<typename T>
	__duel__ const Vec2<T> operator/(const Vec2<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Vec2 operator/ error: n can not be 0!");
		return Vec2<T>(v.x / n, v.y / n);
	}
	template<typename T>
	__duel__ const Vec2<T> operator/(const Vec2<T>& v0, const Vec2<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Vec2 operator/ error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Vec2 operator/ error: v1.y can not be 0!");
		return Vec2<T>(v0.x / v1.x, v0.y / v1.y);
	}

	template<typename T>
	__duel__ const Vec2<T>& operator/=(Vec2<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Vec2 operator/= error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		return v;
	}
	template<typename T>
	__duel__ const Vec2<T>& operator/=(Vec2<T>& v0, const Vec2<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Vec2 operator/= error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Vec2 operator/= error: v1.y can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		return v0;
	}

	__duel__ const Vec2<Int>& operator/=(Vec2<Int>& v, const Float& n) = delete;
	__duel__ const Vec2<Int>& operator/=(Vec2<Int>& v0, const Vec2<Float>& v1) = delete;

#pragma endregion

#pragma endregion

#pragma region Vec2 different type operation

#pragma region Vec2 different type operation +

	template<typename T, typename U>
	__duel__ const Vec2<Float> operator+(const T& n, const Vec2<U>& v)
	{
		return Vec2<Float>(n + v.x, n + v.y);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator+(const Vec2<T>& v, const U& n)
	{
		return Vec2<Float>(v.x + n, v.y + n);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator+(const Vec2<T>& v0, const Vec2<U>& v1)
	{
		return Vec2<Float>(v0.x + v1.x, v0.y + v1.y);
	}

#pragma endregion

#pragma region Vec2 different type operation -

	template<typename T, typename U>
	__duel__ const Vec2<Float> operator-(const T& n, const Vec2<U>& v)
	{
		return Vec2<Float>(n - v.x, n - v.y);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator-(const Vec2<T>& v, const U& n)
	{
		return Vec2<Float>(v.x - n, v.y - n);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator-(const Vec2<T>& v0, const Vec2<U>& v1)
	{
		return Vec2<Float>(v0.x - v1.x, v0.y - v1.y);
	}

#pragma endregion

#pragma region Vec2 different type operation *

	template<typename T, typename U>
	__duel__ const Vec2<Float> operator*(const T& n, const Vec2<U>& v)
	{
		return Vec2<Float>(n * v.x, n * v.y);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator*(const Vec2<T>& v, const U& n)
	{
		return Vec2<Float>(v.x * n, v.y * n);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator*(const Vec2<T>& v0, const Vec2<U>& v1)
	{
		return Vec2<Float>(v0.x * v1.x, v0.y * v1.y);
	}

#pragma endregion

#pragma region Vec2 different type operation /

	template<typename T, typename U>
	__duel__ const Vec2<Float> operator/(const T& n, const Vec2<U>& v)
	{
		CHECK(v.x != 0, "Vec2<Float> operation /(n,Vec2 v): v.x can not be zero.");
		CHECK(v.y != 0, "Vec2<Float> operation /(n,Vec2 v): v.y can not be zero.");
		return Vec2<Float>(n / v.x, n / v.y);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator/(const Vec2<T>& v, const U& n)
	{
		CHECK(v.y != 0, "Vec2<Float> operation /(Vec2 v,n): n can not be zero.");
		return Vec2<Float>(v.x / n, v.y / n);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator/(const Vec2<T>& v0, const Vec2<U>& v1)
	{
		CHECK(v1.x != 0, "Vec2<Float> operation /(Vec2 v0,Vec2 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "Vec2<Float> operation /(Vec2 v0,Vec2 v1): v1.y can not be zero.");
		return Vec2<Float>(v0.x / v1.x, v0.y / v1.y);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const Vec2<T>& v)
	{
		const custd::OStream os;
		os << v.x << "\t" << v.y << custd::endl;
	}

#pragma endregion

#pragma region Vec3
	template<typename T>
	class Vec3
	{
	public:
		T x, y, z;
	public:
		__duel__ Vec3() :x(0), y(0), z(0) {}
		__duel__ Vec3(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}
		__duel__ Vec3(const T& n) : x(n), y(n), z(n) {}
		__duel__ Vec3(const Vec3<T>& v) : x(v.x), y(v.y), z(v.z) {}
		template<typename U>
		__duel__ explicit Vec3(const Vec3<U>& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ ~Vec3() {}
	public:
		__duel__ const Vec3& operator=(const Vec3<int>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "The <idx> in Vec3<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			default: CHECK(false, "Can not run Vec3::operator[idx]: switch::default."); break;
			}
		}
	public:
		const Bool IsZero() const
		{
			return x == 0 && y == 0 && z == 0;
		}
		const Int MaxAbsIdx() const
		{
			T xa = x < 0.0 ? -x : x;
			T ya = y < 0.0 ? -y : y;
			T za = z < 0.0 ? -z : z;
			return xa > ya ? (xa > za ? 0 : 2) : (ya > za ? 1 : 2);
		}
	};

	typedef Vec3<Int> Vec3i;
	typedef Vec3<Float> Vec3f;

#pragma region Vec3 vector operation

	template<typename T>
	__duel__ const Vec3<T> abs(const Vec3<T>& v)
	{
		return Vec3<T>(v.x, v.y, v.z);
	}

	template<typename T>
	__duel__ const T dot(const Vec3<T>& v0, const Vec3<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z;
	}

	template<typename T, typename U>
	__duel__ Float dot(const Vec3<T>& v0, const Vec3<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z;
	}

	template<typename T>
	__duel__ const Float norm(const Vec3<T>& v)
	{
		Float square = v.x*v.x + v.y*v.y + v.z*v.z;
		return sqrt(square);
	}

	template<typename T>
	__duel__ const Vec3<T> cross(const Vec3<T>& v0, const Vec3<T>& v1)
	{
		return Vec3<T>(v0.y*v1.z - v0.z*v1.y, v0.z*v1.x - v0.x*v1.z, v0.x*v1.y - v0.y*v1.x);
	}

	template<typename T, typename U>
	__duel__ const Vec3<Float> cross(const Vec3<T>& v0, const Vec3<U>& v1)
	{
		return Vec3<Float>(v0.y*v1.z - v0.z*v1.y, v0.z*v1.x - v0.x*v1.z, v0.x*v1.y - v0.y*v1.x);
	}

	template<typename T>
	__duel__ const Vec3<Float> normalize(const Vec3<T>& vec)
	{
		Float square = vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
		CHECK(square >= 0.0, "Vec3 normalize error: square can not less than 0.0!");
		Float norm = sqrt(square);
		Float inv = 1.0 / norm;
		return inv * vec;
	}

	template<typename T>
	const Vec3<T> RodriguesRotate(const Vec3<T>& axis,const Float& theta, const Vec3<T>& v)
	{
		if (norm(v) == 0)
			return v;
		auto k = normalize(axis);
		Float sinTheta = sin(theta);
		Float cosTheta = cos(theta);
		return v * cosTheta + cross(k, v)*sinTheta + k * dot(k, v)*(1.0 - cosTheta);
	}

#pragma endregion

#pragma region Vec3 same type operation

#pragma region Vec3 same type operation +

	template<typename T>
	__duel__ const Vec3<T> operator+(const T& n, const Vec3<T>& v)
	{
		return Vec3<T>(n + v.x, n + v.y, n + v.z);
	}
	template<typename T>
	__duel__ const Vec3<T> operator+(const Vec3<T>& v, const T& n)
	{
		return Vec3<T>(v.x + n, v.y + n, v.z + n);
	}
	template<typename T>
	__duel__ const Vec3<T> operator+(const Vec3<T>& v0, const Vec3<T>& v1)
	{
		return Vec3<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	template<typename T, typename U>
	__duel__ const Vec3<T>& operator+=(Vec3<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec3<T>& operator+=(Vec3<T>& v0, const Vec3<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		return v0;
	}

	__duel__ const Vec3<Int>& operator+=(Vec3<Int>& v, const Float& n) = delete;
	__duel__ const Vec3<Int>& operator+=(Vec3<Int>& v0, const Vec3<Float>& v1) = delete;


#pragma endregion

#pragma region Vec3 same type operation -

	template<typename T>
	__duel__ const Vec3<T> operator-(const Vec3<T>& v)
	{
		return Vec3<T>(-v.x, -v.y, -v.z);
	}

	template<typename T>
	__duel__ const Vec3<T> operator-(const T& n, const Vec3<T>& v)
	{
		return Vec3<T>(n - v.x, n - v.y, n - v.z);
	}
	template<typename T>
	__duel__ const Vec3<T> operator-(const Vec3<T>& v, const T& n)
	{
		return Vec3<T>(v.x - n, v.y - n, v.z - n);
	}
	template<typename T>
	__duel__ const Vec3<T> operator-(const Vec3<T>& v0, const Vec3<T>& v1)
	{
		return Vec3<T>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

	template<typename T, typename U>
	__duel__ const Vec3<T>& operator-=(Vec3<T>& v, const U& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec3<T>& operator-=(Vec3<T>& v0, const Vec3<U>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		return v0;
	}

	__duel__ const Vec3<Int>& operator-=(Vec3<Int>& v, const Float& n) = delete;
	__duel__ const Vec3<Int>& operator-=(Vec3<Int>& v0, const Vec3<Float>& v1) = delete;


#pragma endregion

#pragma region Vec3 same type operation *

	template<typename T>
	__duel__ const Vec3<T> operator*(const T& n, const Vec3<T>& v)
	{
		return Vec3<T>(n * v.x, n * v.y, n * v.z);
	}
	template<typename T>
	__duel__ const Vec3<T> operator*(const Vec3<T>& v, const T& n)
	{
		return Vec3<T>(v.x * n, v.y * n, v.z * n);
	}
	template<typename T>
	__duel__ const Vec3<T> operator*(const Vec3<T>& v0, const Vec3<T>& v1)
	{
		return Vec3<T>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	template<typename T, typename U>
	__duel__ const Vec3<T>& operator*=(Vec3<T>& v, const U& n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec3<T>& operator*=(Vec3<T>& v0, const Vec3<U>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		return v0;
	}

	__duel__ const Vec3<Int>& operator*=(Vec3<Int>& v, const Float& n) = delete;
	__duel__ const Vec3<Int>& operator*=(Vec3<Int>& v0, const Vec3<Float>& v1) = delete;


#pragma endregion

#pragma region Vec3 same type operation /

	template<typename T>
	__duel__ const Vec3<T> operator/(const T& n, const Vec3<T>& v)
	{
		CHECK(v.x != 0, "Same type Vec3 operator/(n,Vec3 v) error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type Vec3 operator/(n,Vec3 v) error: v.y can not be 0!");
		CHECK(v.z != 0, "Same type Vec3 operator/(n,Vec3 v) error: v.z can not be 0!");
		return Vec3<T>(n / v.x, n / v.y, n / v.z);
	}
	template<typename T>
	__duel__ const Vec3<T> operator/(const Vec3<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Vec3 operator/(Vec3 v, n) error: n can not be 0!");
		return Vec3<T>(v.x / n, v.y / n, v.z / n);
	}
	template<typename T>
	__duel__ const Vec3<T> operator/(const Vec3<T>& v0, const Vec3<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Vec3 operator/(n,Vec3 v) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Vec3 operator/(n,Vec3 v) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Vec3 operator/(n,Vec3 v) error: v1.z can not be 0!");
		return Vec3<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	template<typename T, typename U>
	__duel__ const Vec3<T>& operator/=(Vec3<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type Vec3 operator/=(Vec3 v, n) error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		v.z /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec3<T>& operator/=(Vec3<T>& v0, const Vec3<U>& v1)
	{
		CHECK(v1.x != 0, "Same type Vec3 operator/=(Vec3 v0,Vec3 v1) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Vec3 operator/=(Vec3 v0,Vec3 v1) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Vec3 operator/=(Vec3 v0,Vec3 v1) error: v1.z can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}

	__duel__ const Vec3<Int>& operator/=(Vec3<Int>& v, const Float& n) = delete;
	__duel__ const Vec3<Int>& operator/=(Vec3<Int>& v0, const Vec3<Float>& v1) = delete;


#pragma endregion


#pragma endregion

#pragma region Vec3 different type operation

#pragma region Vec3 different type operation +

	template<typename T, typename U>
	__duel__ const Vec3<Float> operator+(const T& n, const Vec3<U>& v)
	{
		return Vec3<Float>(n + v.x, n + v.y, n + v.z);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator+(const Vec3<T>& v, const U& n)
	{
		return Vec3<Float>(v.x + n, v.y + n, v.z + n);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator+(const Vec3<T>& v0, const Vec3<U>& v1)
	{
		return Vec3<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

#pragma endregion

#pragma region Vec3 different type operation -

	template<typename T, typename U>
	__duel__ const Vec3<Float> operator-(const T& n, const Vec3<U>& v)
	{
		return Vec3<Float>(n - v.x, n - v.y, n - v.z);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator-(const Vec3<T>& v, const U& n)
	{
		return Vec3<Float>(v.x - n, v.y - n, v.z - n);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator-(const Vec3<T>& v0, const Vec3<U>& v1)
	{
		return Vec3<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

#pragma endregion

#pragma region Vec3 different type operation *

	template<typename T, typename U>
	__duel__ const Vec3<Float> operator*(const T& n, const Vec3<U>& v)
	{
		return Vec3<Float>(n * v.x, n * v.y, n * v.z);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator*(const Vec3<T>& v, const U& n)
	{
		return Vec3<Float>(v.x * n, v.y * n, v.z * n);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator*(const Vec3<T>& v0, const Vec3<U>& v1)
	{
		return Vec3<Float>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

#pragma endregion

#pragma region Vec3 different type operation /

	template<typename T, typename U>
	__duel__ const Vec3<Float> operator/(const T& n, const Vec3<U>& v)
	{
		CHECK(v.x != 0, "Vec3<Float> operation /(n, Vec3 v1): v1.x can not be zero.");
		CHECK(v.y != 0, "Vec3<Float> operation /(n, Vec3 v1): v1.y can not be zero.");
		CHECK(v.z != 0, "Vec3<Float> operation /(n, Vec3 v1): v1.z can not be zero.");
		return Vec3<Float>(n / v.x, n / v.y, n / v.z);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator/(const Vec3<T>& v, const U& n)
	{
		CHECK(n != 0, "Vec3<Float> operation /(Vec3 v, n): n can not be zero.");
		return Vec3<Float>(v.x / n, v.y / n, v.z / n);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator/(const Vec3<T>& v0, const Vec3<U>& v1)
	{
		CHECK(v1.x != 0, "Vec3<Float> operation /(Vec3 v0, Vec3 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "Vec3<Float> operation /(Vec3 v0, Vec3 v1): v1.y can not be zero.");
		CHECK(v1.z != 0, "Vec3<Float> operation /(Vec3 v0, Vec3 v1): v1.z can not be zero.");
		return Vec3<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const Vec3<T>& v)
	{
		const custd::OStream os;
		os << v.x << "\t" << v.y << "\t" << v.z << custd::endl;
	}

#pragma endregion

#pragma region Point2
	template<typename T>
	class Point2
	{
	public:
		T x, y;
	public:
		__duel__ Point2() :x(0), y(0) {}
		__duel__ Point2(const T& n) : x(n), y(n) {}
		__duel__ Point2(const T& _x, const T& _y) : x(_x), y(_y) {}
		__duel__ Point2(const Point2<T>& v) : x(v.x), y(v.y) {}
		template<typename U>
		__duel__ explicit Point2(const Point2<U>& v) : x(v.x), y(v.y) {}
		__duel__ ~Point2() {}
	public:
		__duel__ const Point2& operator=(const Point2<int>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 1, "The <idx> in Point2<T>::operator[idx] is illegal!");
			return idx == 0 ? x : y;
		}
	};

	typedef Point2<Int> Point2i;
	typedef Point2<Float> Point2f;

#pragma region Point2 same type operation

#pragma region Point2 same type operation +

	template<typename T>
	__duel__ const Point2<T> operator+(const T& n, const Point2<T>& v)
	{
		return Point2<T>(n + v.x, n + v.y);
	}
	template<typename T>
	__duel__ const Point2<T> operator+(const Point2<T>& v, const T& n)
	{
		return Point2<T>(v.x + n, v.y + n);
	}
	template<typename T>
	__duel__ const Point2<T> operator+(const Point2<T>& v0, const Point2<T>& v1)
	{
		return Point2<T>(v0.x + v1.x, v0.y + v1.y);
	}

	template<typename T, typename U>
	__duel__ const Point2<T>& operator+=(Point2<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Point2<T>& operator+=(Point2<T>& v0, const Point2<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		return v0;
	}

	__duel__ const Point2<Int>& operator+=(Point2<Int>& v, const Float& n) = delete;
	__duel__ const Point2<Int>& operator+=(Point2<Int>& v0, const Point2<Float>& v1) = delete;


#pragma endregion

#pragma region Point2 same type operation -

	template<typename T>
	__duel__ const Vec2<T> operator-(const T& n, const Point2<T>& v)
	{
		return Vec2<T>(n - v.x, n - v.y);
	}
	template<typename T>
	__duel__ const Vec2<T> operator-(const Point2<T>& v, const T& n)
	{
		return Vec2<T>(v.x - n, v.y - n);
	}
	template<typename T>
	__duel__ const Vec2<T> operator-(const Point2<T>& v0, const Point2<T>& v1)
	{
		return Vec2<T>(v0.x - v1.x, v0.y - v1.y);
	}

	template<typename T>
	__duel__ const Vec2<T>& operator-=(Point2<T>& v, const T& n)
	{
		v.x -= n;
		v.y -= n;
		return v;
	}
	template<typename T>
	__duel__ const Vec2<T>& operator-=(Point2<T>& v0, const Point2<T>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		return v0;
	}

	__duel__ const Vec2<Int>& operator-=(Point2<Int>& v, const Float& n) = delete;
	__duel__ const Vec2<Int>& operator-=(Point2<Int>& v0, const Point2<Float>& v1) = delete;

#pragma endregion

#pragma region Point2 same type operation *

	template<typename T>
	__duel__ const Point2<T> operator*(const T& n, const Point2<T>& v)
	{
		return Point2<T>(n * v.x, n * v.y);
	}
	template<typename T>
	__duel__ const Point2<T> operator*(const Point2<T>& v, const T& n)
	{
		return Point2<T>(v.x * n, v.y * n);
	}
	template<typename T>
	__duel__ const Point2<T> operator*(const Point2<T>& v0, const Point2<T>& v1)
	{
		return Point2<T>(v0.x * v1.x, v0.y * v1.y);
	}

	template<typename T>
	__duel__ const Point2<T>& operator*=(Point2<T>& v, const T& n)
	{
		v.x *= n;
		v.y *= n;
		return v;
	}
	template<typename T>
	__duel__ const Point2<T>& operator*=(Point2<T>& v0, const Point2<T>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		return v0;
	}

	__duel__ const Point2<Int>& operator*=(Point2<Int>& v, const Float& n) = delete;
	__duel__ const Point2<Int>& operator*=(Point2<Int>& v0, const Point2<Float>& v1) = delete;

#pragma endregion

#pragma region Point2 same type operation /

	template<typename T>
	__duel__ const Point2<T> operator/(const T& n, const Point2<T>& v)
	{
		CHECK(v.x != 0, "Same type Point2 operator/ error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type Point2 operator/ error: v.y can not be 0!");
		return Point2<T>(n / v.x, n / v.y);
	}
	template<typename T>
	__duel__ const Point2<T> operator/(const Point2<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Point2 operator/ error: n can not be 0!");
		return Point2<T>(v.x / n, v.y / n);
	}
	template<typename T>
	__duel__ const Point2<T> operator/(const Point2<T>& v0, const Point2<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Point2 operator/ error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Point2 operator/ error: v1.y can not be 0!");
		return Point2<T>(v0.x / v1.x, v0.y / v1.y);
	}

	template<typename T>
	__duel__ const Point2<T>& operator/=(Point2<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Point2 operator/= error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		return v;
	}
	template<typename T>
	__duel__ const Point2<T>& operator/=(Point2<T>& v0, const Point2<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Point2 operator/= error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Point2 operator/= error: v1.y can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		return v0;
	}

	__duel__ const Point2<Int>& operator/=(Point2<Int>& v, const Float& n) = delete;
	__duel__ const Point2<Int>& operator/=(Point2<Int>& v0, const Point2<Float>& v1) = delete;

#pragma endregion

#pragma endregion

#pragma region Point2 different type operation

#pragma region Point2 different type operation +

	template<typename T, typename U>
	__duel__ const Point2<Float> operator+(const T& n, const Point2<U>& v)
	{
		return Point2<Float>(n + v.x, n + v.y);
	}
	template<typename T, typename U>
	__duel__ const Point2<Float> operator+(const Point2<T>& v, const U& n)
	{
		return Point2<Float>(v.x + n, v.y + n);
	}
	template<typename T, typename U>
	__duel__ const Point2<Float> operator+(const Point2<T>& v0, const Point2<U>& v1)
	{
		return Point2<Float>(v0.x + v1.x, v0.y + v1.y);
	}

#pragma endregion

#pragma region Point2 different type operation -

	template<typename T, typename U>
	__duel__ const Vec2<Float> operator-(const T& n, const Point2<U>& v)
	{
		return Vec2<Float>(n - v.x, n - v.y);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator-(const Point2<T>& v, const U& n)
	{
		return Vec2<Float>(v.x - n, v.y - n);
	}
	template<typename T, typename U>
	__duel__ const Vec2<Float> operator-(const Point2<T>& v0, const Point2<U>& v1)
	{
		return Vec2<Float>(v0.x - v1.x, v0.y - v1.y);
	}

#pragma endregion

#pragma region Point2 different type operation *

	template<typename T, typename U>
	__duel__ const Point2<Float> operator*(const T& n, const Point2<U>& v)
	{
		return Point2<Float>(n * v.x, n * v.y);
	}
	template<typename T, typename U>
	__duel__ const Point2<Float> operator*(const Point2<T>& v, const U& n)
	{
		return Point2<Float>(v.x * n, v.y * n);
	}
	template<typename T, typename U>
	__duel__ const Point2<Float> operator*(const Point2<T>& v0, const Point2<U>& v1)
	{
		return Point2<Float>(v0.x * v1.x, v0.y * v1.y);
	}

#pragma endregion

#pragma region Point2 different type operation /

	template<typename T, typename U>
	__duel__ const Point2<Float> operator/(const T& n, const Point2<U>& v)
	{
		CHECK(v.x != 0, "Point2<Float> operation /(n,Point2 v): v.x can not be zero.");
		CHECK(v.y != 0, "Point2<Float> operation /(n,Point2 v): v.y can not be zero.");
		return Point2<Float>(n / v.x, n / v.y);
	}
	template<typename T, typename U>
	__duel__ const Point2<Float> operator/(const Point2<T>& v, const U& n)
	{
		CHECK(v.y != 0, "Point2<Float> operation /(Point2 v,n): n can not be zero.");
		return Point2<Float>(v.x / n, v.y / n);
	}
	template<typename T, typename U>
	__duel__ const Point2<Float> operator/(const Point2<T>& v0, const Point2<U>& v1)
	{
		CHECK(v1.x != 0, "Point2<Float> operation /(Point2 v0,Point2 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "Point2<Float> operation /(Point2 v0,Point2 v1): v1.y can not be zero.");
		return Point2<Float>(v0.x / v1.x, v0.y / v1.y);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const Point2<T>& v)
	{
		const custd::OStream os;
		os << v.x << "\t" << v.y << custd::endl;
	}

#pragma endregion

#pragma region Point3
	template<typename T>
	class Point3
	{
	public:
		T x, y, z;
	public:
		__duel__ Point3() :x(0), y(0), z(0) {}
		__duel__ Point3(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}
		__duel__ Point3(const T& n) : x(n), y(n), z(n) {}
		__duel__ Point3(const Point3<T>& v) : x(v.x), y(v.y), z(v.z) {}
		template<typename U>
		__duel__ explicit Point3(const Point3<U>& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ ~Point3() {}
	public:
		__duel__ const Point3& operator=(const Point3<int>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}


		__duel__ const Point3(const Vec3<T>& v)
			: x(v.x), y(v.y), z(v.z)
		{

		}

		__duel__ const Point3& operator=(const Vec3<T>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "The <idx> in Point3<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			default: CHECK(false, "Can not run Point3::operator[idx]: switch::default."); break;
			}
		}
	};

	typedef Point3<Int> Point3i;
	typedef Point3<Float> Point3f;

#pragma region Point3 vector operation

	template<typename T>
	__duel__ const T dot(const Point3<T>& v0, const Point3<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z;
	}

	template<typename T, typename U>
	__duel__ Float dot(const Point3<T>& v0, const Point3<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z;
	}

	template<typename T>
	__duel__ const Point3<T> cross(const Point3<T>& v0, const Point3<T>& v1)
	{
		return Point3<T>(v0.y*v1.z - v0.z*v1.y, v0.z*v1.x - v0.x*v1.z, v0.x*v1.y - v0.y*v1.x);
	}

	template<typename T, typename U>
	__duel__ const Point3<Float> cross(const Point3<T>& v0, const Point3<U>& v1)
	{
		return Point3<Float>(v0.y*v1.z - v0.z*v1.y, v0.z*v1.x - v0.x*v1.z, v0.x*v1.y - v0.y*v1.x);
	}

	template<typename T>
	__duel__ const Point3<Float> normalize(const Point3<T>& vec)
	{
		Float square = vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
		CHECK(square > 0.0, "Point3 normalize error: square can not less than 0.0!");
		Float norm = sqrt(square);
		Float inv = 1.0 / norm;
		return inv * vec;
	}

#pragma endregion

#pragma region Point3 same type operation

#pragma region Point3 same type operation +

	template<typename T>
	__duel__ const Point3<T> operator+(const T& n, const Point3<T>& v)
	{
		return Point3<T>(n + v.x, n + v.y, n + v.z);
	}
	template<typename T>
	__duel__ const Point3<T> operator+(const Point3<T>& v, const T& n)
	{
		return Point3<T>(v.x + n, v.y + n, v.z + n);
	}
	template<typename T>
	__duel__ const Point3<T> operator+(const Point3<T>& v0, const Point3<T>& v1)
	{
		return Point3<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	template<typename T, typename U>
	__duel__ const Point3<T>& operator+=(Point3<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Point3<T>& operator+=(Point3<T>& v0, const Point3<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		return v0;
	}

	__duel__ const Point3<Int>& operator+=(Point3<Int>& v, const Float& n) = delete;
	__duel__ const Point3<Int>& operator+=(Point3<Int>& v0, const Point3<Float>& v1) = delete;


#pragma endregion

#pragma region Point3 same type operation -

	template<typename T>
	__duel__ const Vec3<T> operator-(const T& n, const Point3<T>& v)
	{
		return Vec3<T>(n - v.x, n - v.y, n - v.z);
	}
	template<typename T>
	__duel__ const Vec3<T> operator-(const Point3<T>& v, const T& n)
	{
		return Vec3<T>(v.x - n, v.y - n, v.z - n);
	}
	template<typename T>
	__duel__ const Vec3<T> operator-(const Point3<T>& p0, const Point3<T>& p1)
	{
		return Vec3<T>(p0.x - p1.x, p0.y - p1.y, p0.z - p1.z);
	}

	template<typename T, typename U>
	__duel__ const Point3<T>& operator-=(Point3<T>& v, const U& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Point3<T>& operator-=(Point3<T>& v0, const Point3<U>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		return v0;
	}

	__duel__ const Point3<Int>& operator-=(Point3<Int>& v, const Float& n) = delete;
	__duel__ const Point3<Int>& operator-=(Point3<Int>& v0, const Point3<Float>& v1) = delete;


#pragma endregion

#pragma region Point3 same type operation *

	template<typename T>
	__duel__ const Point3<T> operator*(const T& n, const Point3<T>& v)
	{
		return Point3<T>(n * v.x, n * v.y, n * v.z);
	}
	template<typename T>
	__duel__ const Point3<T> operator*(const Point3<T>& v, const T& n)
	{
		return Point3<T>(v.x * n, v.y * n, v.z * n);
	}
	template<typename T>
	__duel__ const Point3<T> operator*(const Point3<T>& v0, const Point3<T>& v1)
	{
		return Point3<T>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	template<typename T, typename U>
	__duel__ const Point3<T>& operator*=(Point3<T>& v, const U& n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Point3<T>& operator*=(Point3<T>& v0, const Point3<U>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		return v0;
	}

	__duel__ const Point3<Int>& operator*=(Point3<Int>& v, const Float& n) = delete;
	__duel__ const Point3<Int>& operator*=(Point3<Int>& v0, const Point3<Float>& v1) = delete;


#pragma endregion

#pragma region Point3 same type operation /

	template<typename T>
	__duel__ const Point3<T> operator/(const T& n, const Point3<T>& v)
	{
		CHECK(v.x != 0, "Same type Point3 operator/(n,Point3 v) error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type Point3 operator/(n,Point3 v) error: v.y can not be 0!");
		CHECK(v.z != 0, "Same type Point3 operator/(n,Point3 v) error: v.z can not be 0!");
		return Point3<T>(n / v.x, n / v.y, n / v.z);
	}
	template<typename T>
	__duel__ const Point3<T> operator/(const Point3<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Point3 operator/(Point3 v, n) error: n can not be 0!");
		return Point3<T>(v.x / n, v.y / n, v.z / n);
	}
	template<typename T>
	__duel__ const Point3<T> operator/(const Point3<T>& v0, const Point3<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Point3 operator/(n,Point3 v) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Point3 operator/(n,Point3 v) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Point3 operator/(n,Point3 v) error: v1.z can not be 0!");
		return Point3<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	template<typename T, typename U>
	__duel__ const Point3<T>& operator/=(Point3<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type Point3 operator/=(Point3 v, n) error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		v.z /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Point3<T>& operator/=(Point3<T>& v0, const Point3<U>& v1)
	{
		CHECK(v1.x != 0, "Same type Point3 operator/=(Point3 v0,Point3 v1) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Point3 operator/=(Point3 v0,Point3 v1) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Point3 operator/=(Point3 v0,Point3 v1) error: v1.z can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}

	__duel__ const Point3<Int>& operator/=(Point3<Int>& v, const Float& n) = delete;
	__duel__ const Point3<Int>& operator/=(Point3<Int>& v0, const Point3<Float>& v1) = delete;


#pragma endregion


#pragma endregion

#pragma region Point3 different type operation

#pragma region Point3 different type operation +

	template<typename T, typename U>
	__duel__ const Point3<Float> operator+(const T& n, const Point3<U>& v)
	{
		return Point3<Float>(n + v.x, n + v.y, n + v.z);
	}
	template<typename T, typename U>
	__duel__ const Point3<Float> operator+(const Point3<T>& v, const U& n)
	{
		return Point3<Float>(v.x + n, v.y + n, v.z + n);
	}
	template<typename T, typename U>
	__duel__ const Point3<Float> operator+(const Point3<T>& v0, const Point3<U>& v1)
	{
		return Point3<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

#pragma endregion

#pragma region Point3 different type operation -

	template<typename T, typename U>
	__duel__ const Vec3<Float> operator-(const T& n, const Point3<U>& v)
	{
		return Vec3<Float>(n - v.x, n - v.y, n - v.z);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator-(const Point3<T>& v, const U& n)
	{
		return Vec3<Float>(v.x - n, v.y - n, v.z - n);
	}
	template<typename T, typename U>
	__duel__ const Vec3<Float> operator-(const Point3<T>& v0, const Point3<U>& v1)
	{
		return Vec3<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

#pragma endregion

#pragma region Point3 different type operation *

	template<typename T, typename U>
	__duel__ const Point3<Float> operator*(const T& n, const Point3<U>& v)
	{
		return Point3<Float>(n * v.x, n * v.y, n * v.z);
	}
	template<typename T, typename U>
	__duel__ const Point3<Float> operator*(const Point3<T>& v, const U& n)
	{
		return Point3<Float>(v.x * n, v.y * n, v.z * n);
	}
	template<typename T, typename U>
	__duel__ const Point3<Float> operator*(const Point3<T>& v0, const Point3<U>& v1)
	{
		return Point3<Float>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

#pragma endregion

#pragma region Point3 different type operation /

	template<typename T, typename U>
	__duel__ const Point3<Float> operator/(const T& n, const Point3<U>& v)
	{
		CHECK(v.x != 0, "Point3<Float> operation /(n, Point3 v1): v1.x can not be zero.");
		CHECK(v.y != 0, "Point3<Float> operation /(n, Point3 v1): v1.y can not be zero.");
		CHECK(v.z != 0, "Point3<Float> operation /(n, Point3 v1): v1.z can not be zero.");
		return Point3<Float>(n / v.x, n / v.y, n / v.z);
	}
	template<typename T, typename U>
	__duel__ const Point3<Float> operator/(const Point3<T>& v, const U& n)
	{
		CHECK(n != 0, "Point3<Float> operation /(Point3 v, n): n can not be zero.");
		return Point3<Float>(v.x / n, v.y / n, v.z / n);
	}
	template<typename T, typename U>
	__duel__ const Point3<Float> operator/(const Point3<T>& v0, const Point3<U>& v1)
	{
		CHECK(v1.x != 0, "Point3<Float> operation /(Point3 v0, Point3 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "Point3<Float> operation /(Point3 v0, Point3 v1): v1.y can not be zero.");
		CHECK(v1.z != 0, "Point3<Float> operation /(Point3 v0, Point3 v1): v1.z can not be zero.");
		return Point3<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const Point3<T>& v)
	{
		const custd::OStream os;
		os << v.x << "\t" << v.y << "\t" << v.z << custd::endl;
	}

#pragma endregion

#pragma region Normal

	template<typename T>
	class Normal3
	{
	public:
		T x, y, z;
	public:
		__duel__ Normal3() : x(0), y(1), z(0) {}
		__duel__ Normal3(const T& _x, const T& _y, const T& _z)
		{
			Vec3<Float> normal = normalize(CUM::Vec3<T>(_x, _y, _z));
			x = normal.x;
			y = normal.y;
			z = normal.z;
		}
		__duel__ Normal3(const Vec3<T>& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ const Normal3& operator=(const Vec3<T>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
	};

	typedef Normal3<Float> Normal3f;
#pragma endregion

#pragma region Vec4
	template<typename T>
	class Vec4
	{
	public:
		T x, y, z, w;
	public:
		__duel__ Vec4() :x(0), y(0), z(0), w(0) {}
		__duel__ Vec4(const T& _x, const T& _y, const T& _z, const T& _w) : x(_x), y(_y), z(_z), w(_w) {}
		__duel__ Vec4(const T& n) : x(n), y(n), z(n), w(n) {}
		__duel__ Vec4(const Vec4<T>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		template<typename U>
		__duel__ explicit Vec4(const Vec4<U>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		__duel__ ~Vec4() {}
	public:
		__duel__ const Vec4& operator=(const Vec4<int>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			w = v.w;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 3, "The <idx> in Vec4<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			case 3: return w; break;
			default: CHECK(false, "Can not run Vec4::operator[idx]: switch::default."); break;
			}
		}
	};

	typedef Vec4<Int> Vec4i;
	typedef Vec4<Float> Vec4f;

#pragma region Vec4 vector operation

	template<typename T>
	__duel__ const T dot(const Vec4<T>& v0, const Vec4<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z + v0.w * v1.w;
	}

	template<typename T, typename U>
	__duel__ Float dot(const Vec4<T>& v0, const Vec4<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z + v0.w*v1.w;
	}

	template<typename T>
	const Float norm(const Vec4<T>& v)
	{
		Float square = 0.0;
		for (Int i = 0; i < 4; i++)
		{
			square += v[i] * v[i];
		}
		return sqrt(square);
	}

	template<typename T>
	__duel__ const Vec4<Float> normalize(const Vec4<T>& vec)
	{
		Float square = vec.x*vec.x + vec.y*vec.y + vec.z*vec.z + vec.w*vec.w;
		CHECK(square > 0.0, "Vec4 normalize error: square can not less than 0.0!");
		Float norm = sqrt(square);
		Float inv = 1.0 / norm;
		return inv * vec;
	}

#pragma endregion

#pragma region Vec4 same type operation

#pragma region Vec4 same type operation +

	template<typename T>
	__duel__ const Vec4<T> operator+(const T& n, const Vec4<T>& v)
	{
		return Vec4<T>(n + v.x, n + v.y, n + v.z, n + v.w);
	}
	template<typename T>
	__duel__ const Vec4<T> operator+(const Vec4<T>& v, const T& n)
	{
		return Vec4<T>(v.x + n, v.y + n, v.z + n, v.w + n);
	}
	template<typename T>
	__duel__ const Vec4<T> operator+(const Vec4<T>& v0, const Vec4<T>& v1)
	{
		return Vec4<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
	}

	template<typename T, typename U>
	__duel__ const Vec4<T>& operator+=(Vec4<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		v.w += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec4<T>& operator+=(Vec4<T>& v0, const Vec4<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		v0.w += v1.w;
		return v0;
	}

	__duel__ const Vec4<Int>& operator+=(Vec4<Int>& v, const Float& n) = delete;
	__duel__ const Vec4<Int>& operator+=(Vec4<Int>& v0, const Vec4<Float>& v1) = delete;


#pragma endregion

#pragma region Vec4 same type operation -

	template<typename T>
	__duel__ const Vec4<T> operator-(const T& n, const Vec4<T>& v)
	{
		return Vec4<T>(n - v.x, n - v.y, n - v.z, n - v.w);
	}
	template<typename T>
	__duel__ const Vec4<T> operator-(const Vec4<T>& v, const T& n)
	{
		return Vec4<T>(v.x - n, v.y - n, v.z - n, v.w - n);
	}
	template<typename T>
	__duel__ const Vec4<T> operator-(const Vec4<T>& v0, const Vec4<T>& v1)
	{
		return Vec4<T>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
	}

	template<typename T, typename U>
	__duel__ const Vec4<T>& operator-=(Vec4<T>& v, const U& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		v.w -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec4<T>& operator-=(Vec4<T>& v0, const Vec4<U>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		v0.w -= v1.w;
		return v0;
	}

	__duel__ const Vec4<Int>& operator-=(Vec4<Int>& v, const Float& n) = delete;
	__duel__ const Vec4<Int>& operator-=(Vec4<Int>& v0, const Vec4<Float>& v1) = delete;


#pragma endregion

#pragma region Vec4 same type operation *

	template<typename T>
	__duel__ const Vec4<T> operator*(const T& n, const Vec4<T>& v)
	{
		return Vec4<T>(n * v.x, n * v.y, n * v.z, n * v.w);
	}
	template<typename T>
	__duel__ const Vec4<T> operator*(const Vec4<T>& v, const T& n)
	{
		return Vec4<T>(v.x * n, v.y * n, v.z * n,v.w * n);
	}
	template<typename T>
	__duel__ const Vec4<T> operator*(const Vec4<T>& v0, const Vec4<T>& v1)
	{
		return Vec4<T>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
	}

	template<typename T, typename U>
	__duel__ const Vec4<T>& operator*=(Vec4<T>& v, const U& n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		v.w *= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec4<T>& operator*=(Vec4<T>& v0, const Vec4<U>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		v0.w *= v1.w;
		return v0;
	}

	__duel__ const Vec4<Int>& operator*=(Vec4<Int>& v, const Float& n) = delete;
	__duel__ const Vec4<Int>& operator*=(Vec4<Int>& v0, const Vec4<Float>& v1) = delete;


#pragma endregion

#pragma region Vec4 same type operation /

	template<typename T>
	__duel__ const Vec4<T> operator/(const T& n, const Vec4<T>& v)
	{
		CHECK(v.x != 0, "Same type Vec4 operator/(n,Vec4 v) error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type Vec4 operator/(n,Vec4 v) error: v.y can not be 0!");
		CHECK(v.z != 0, "Same type Vec4 operator/(n,Vec4 v) error: v.z can not be 0!");
		CHECK(v.w != 0, "Same type Vec4 operator/(n,Vec4 v) error: v.w can not be 0!");
		return Vec4<T>(n / v.x, n / v.y, n / v.z, n / v.w);
	}
	template<typename T>
	__duel__ const Vec4<T> operator/(const Vec4<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Vec4 operator/(Vec4 v, n) error: n can not be 0!");
		return Vec4<T>(v.x / n, v.y / n, v.z / n, v.w / n);
	}
	template<typename T>
	__duel__ const Vec4<T> operator/(const Vec4<T>& v0, const Vec4<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Vec4 operator/(n,Vec4 v) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Vec4 operator/(n,Vec4 v) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Vec4 operator/(n,Vec4 v) error: v1.z can not be 0!");
		CHECK(v1.w != 0, "Same type Vec4 operator/(n,Vec4 v) error: v1.w can not be 0!");
		return Vec4<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	template<typename T, typename U>
	__duel__ const Vec4<T>& operator/=(Vec4<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type Vec4 operator/=(Vec4 v, n) error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		v.z /= n;
		v.w /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Vec4<T>& operator/=(Vec4<T>& v0, const Vec4<U>& v1)
	{
		CHECK(v1.x != 0, "Same type Vec4 operator/=(Vec4 v0,Vec4 v1) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Vec4 operator/=(Vec4 v0,Vec4 v1) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Vec4 operator/=(Vec4 v0,Vec4 v1) error: v1.z can not be 0!");
		CHECK(v1.w != 0, "Same type Vec4 operator/=(Vec4 v0,Vec4 v1) error: v1.w can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		v0.w /= v1.w;
		return v0;
	}

	__duel__ const Vec4<Int>& operator/=(Vec4<Int>& v, const Float& n) = delete;
	__duel__ const Vec4<Int>& operator/=(Vec4<Int>& v0, const Vec4<Float>& v1) = delete;


#pragma endregion


#pragma endregion

#pragma region Vec4 different type operation

#pragma region Vec4 different type operation +

	template<typename T, typename U>
	__duel__ const Vec4<Float> operator+(const T& n, const Vec4<U>& v)
	{
		return Vec4<Float>(n + v.x, n + v.y, n+ v.z, n + v.w);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator+(const Vec4<T>& v, const U& n)
	{
		return Vec4<Float>(v.x + n, v.y + n, v.z + n, v.w + n);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator+(const Vec4<T>& v0, const Vec4<U>& v1)
	{
		return Vec4<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
	}

#pragma endregion

#pragma region Vec4 different type operation -

	template<typename T, typename U>
	__duel__ const Vec4<Float> operator-(const T& n, const Vec4<U>& v)
	{
		return Vec4<Float>(n - v.x, n - v.y, n - v.z, n - v.w);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator-(const Vec4<T>& v, const U& n)
	{
		return Vec4<Float>(v.x - n, v.y - n, v.z - n, v.w - n);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator-(const Vec4<T>& v0, const Vec4<U>& v1)
	{
		return Vec4<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
	}

#pragma endregion

#pragma region Vec4 different type operation *

	template<typename T, typename U>
	__duel__ const Vec4<Float> operator*(const T& n, const Vec4<U>& v)
	{
		return Vec4<Float>(n * v.x, n * v.y, n * v.z, n * v.w);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator*(const Vec4<T>& v, const U& n)
	{
		return Vec4<Float>(v.x * n, v.y * n, v.z * n, v.w * n);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator*(const Vec4<T>& v0, const Vec4<U>& v1)
	{
		return Vec4<Float>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
	}

#pragma endregion

#pragma region Vec4 different type operation /

	template<typename T, typename U>
	__duel__ const Vec4<Float> operator/(const T& n, const Vec4<U>& v)
	{
		CHECK(v.x != 0, "Vec4<Float> operation /(n, Vec4 v1): v1.x can not be zero.");
		CHECK(v.y != 0, "Vec4<Float> operation /(n, Vec4 v1): v1.y can not be zero.");
		CHECK(v.z != 0, "Vec4<Float> operation /(n, Vec4 v1): v1.z can not be zero.");
		CHECK(v.w != 0, "Vec4<Float> operation /(n, Vec4 v1): v1.w can not be zero.");
		return Vec4<Float>(n / v.x, n / v.y, n / v.z, n / v.w);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator/(const Vec4<T>& v, const U& n)
	{
		CHECK(n != 0, "Vec4<Float> operation /(Vec4 v, n): n can not be zero.");
		return Vec4<Float>(v.x / n, v.y / n, v.z / n, v.w / n);
	}
	template<typename T, typename U>
	__duel__ const Vec4<Float> operator/(const Vec4<T>& v0, const Vec4<U>& v1)
	{
		CHECK(v1.x != 0, "Vec4<Float> operation /(Vec4 v0, Vec4 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "Vec4<Float> operation /(Vec4 v0, Vec4 v1): v1.y can not be zero.");
		CHECK(v1.z != 0, "Vec4<Float> operation /(Vec4 v0, Vec4 v1): v1.z can not be zero.");
		CHECK(v1.w != 0, "Vec4<Float> operation /(Vec4 v0, Vec4 v1): v1.w can not be zero.");
		return Vec4<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const Vec4<T>& v)
	{
		const custd::OStream os;
		os << v.x << "\t" << v.y << "\t" << v.z << "\t" << v.w << custd::endl;
	}

#pragma endregion

#pragma region Quaternion
	template<typename T>
	class Quaternion
	{
	public:
		T x, y, z, w;
	public:
		__duel__ Quaternion() :x(0), y(0), z(0), w(0) {}
		__duel__ Quaternion(const T& _x, const T& _y, const T& _z, const T& _w) : x(_x), y(_y), z(_z), w(_w) {}
		__duel__ Quaternion(const T& n) : x(n), y(n), z(n), w(n) {}
		__duel__ Quaternion(const Quaternion<T>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		__duel__ Quaternion(const Vec3<T>& v, const T& _w, const Bool& isRotate = true) : x(v.x), y(v.y), z(v.z), w(_w)
		{
			if (isRotate)
			{
				Vec3<T> axis = normalize(Vec3<T>(v.x, v.y, v.z));
				T theta = _w;
				T sinThetaDi2 = sin(0.5 * theta);
				T cosThetaDi2 = cos(0.5 * theta);
				axis = sinThetaDi2 * axis;
				x = axis.x;
				y = axis.y;
				z = axis.z;
				w = cosThetaDi2;
			}
		}
		template<typename U>
		__duel__ explicit Quaternion(const Quaternion<U>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		__duel__ ~Quaternion() {}
	public:
		__duel__ const Quaternion& operator=(const Quaternion<int>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			w = v.w;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 3, "The <idx> in Quaternion<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			case 3: return w; break;
			default: CHECK(false, "Can not run Quaternion::operator[idx]: switch::default."); break;
			}
		}
	};

	typedef Quaternion<Int> Quaternioni;
	typedef Quaternion<Float> Quaternionf;

#pragma region Quaternion same type operation

#pragma region Quaternion same type operation +

	template<typename T>
	__duel__ const Quaternion<T> operator+(const T& n, const Quaternion<T>& v)
	{
		return Quaternion<T>(n + v.x, n + v.y, n + v.z, n + v.w);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator+(const Quaternion<T>& v, const T& n)
	{
		return Quaternion<T>(v.x + n, v.y + n, v.z + n, v.w + n);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator+(const Quaternion<T>& v0, const Quaternion<T>& v1)
	{
		return Quaternion<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
	}

	template<typename T, typename U>
	__duel__ const Quaternion<T>& operator+=(Quaternion<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		v.w += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Quaternion<T>& operator+=(Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		v0.w += v1.w;
		return v0;
	}

	__duel__ const Quaternion<Int>& operator+=(Quaternion<Int>& v, const Float& n) = delete;
	__duel__ const Quaternion<Int>& operator+=(Quaternion<Int>& v0, const Quaternion<Float>& v1) = delete;


#pragma endregion

#pragma region Quaternion same type operation -

	template<typename T>
	__duel__ const Quaternion<T> operator-(const T& n, const Quaternion<T>& v)
	{
		return Quaternion<T>(n - v.x, n - v.y, n - v.z, n - v.w);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator-(const Quaternion<T>& v, const T& n)
	{
		return Quaternion<T>(v.x - n, v.y - n, v.z - n, v.w - n);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator-(const Quaternion<T>& v0, const Quaternion<T>& v1)
	{
		return Quaternion<T>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
	}

	template<typename T, typename U>
	__duel__ const Quaternion<T>& operator-=(Quaternion<T>& v, const U& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		v.w -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Quaternion<T>& operator-=(Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		v0.w -= v1.w;
		return v0;
	}

	__duel__ const Quaternion<Int>& operator-=(Quaternion<Int>& v, const Float& n) = delete;
	__duel__ const Quaternion<Int>& operator-=(Quaternion<Int>& v0, const Quaternion<Float>& v1) = delete;


#pragma endregion

#pragma region Quaternion same type operation *

	template<typename T>
	__duel__ const Quaternion<T> operator*(const T& n, const Quaternion<T>& v)
	{

		return Quaternion<T>(n * v.x, n * v.y, n * v.z, n * v.w);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator*(const Quaternion<T>& v, const T& n)
	{
		return Quaternion<T>(v.x * n, v.y * n, v.z * n, v.w * n);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator*(const Quaternion<T>& v0, const Quaternion<T>& v1)
	{
		T a = v0.w;
		T b = v1.w;
		Vec3<T> u(v0.x, v0.y, v0.z);
		Vec3<T> v(v1.x, v1.y, v1.z);
		auto ur = a * v + b * u + cross(u, v);
		T w = a * b - dot(u, v);
		return Quaternion<T>(ur, w);
	}
#pragma endregion

#pragma region Quaternion same type operation /

	template<typename T>
	__duel__ const Quaternion<T> operator/(const T& n, const Quaternion<T>& v)
	{
		CHECK(v.x != 0, "Same type Quaternion operator/(n,Quaternion v) error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type Quaternion operator/(n,Quaternion v) error: v.y can not be 0!");
		CHECK(v.z != 0, "Same type Quaternion operator/(n,Quaternion v) error: v.z can not be 0!");
		CHECK(v.w != 0, "Same type Quaternion operator/(n,Quaternion v) error: v.w can not be 0!");
		return Quaternion<T>(n / v.x, n / v.y, n / v.z, n / v.w);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator/(const Quaternion<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Quaternion operator/(Quaternion v, n) error: n can not be 0!");
		return Quaternion<T>(v.x / n, v.y / n, v.z / n, v.w / n);
	}
	template<typename T>
	__duel__ const Quaternion<T> operator/(const Quaternion<T>& v0, const Quaternion<T>& v1)
	{
		CHECK(v1.x != 0, "Same type Quaternion operator/(n,Quaternion v) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Quaternion operator/(n,Quaternion v) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Quaternion operator/(n,Quaternion v) error: v1.z can not be 0!");
		CHECK(v1.w != 0, "Same type Quaternion operator/(n,Quaternion v) error: v1.w can not be 0!");
		return Quaternion<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	template<typename T, typename U>
	__duel__ const Quaternion<T>& operator/=(Quaternion<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type Quaternion operator/=(Quaternion v, n) error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		v.z /= n;
		v.w /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Quaternion<T>& operator/=(Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		CHECK(v1.x != 0, "Same type Quaternion operator/=(Quaternion v0,Quaternion v1) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type Quaternion operator/=(Quaternion v0,Quaternion v1) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type Quaternion operator/=(Quaternion v0,Quaternion v1) error: v1.z can not be 0!");
		CHECK(v1.w != 0, "Same type Quaternion operator/=(Quaternion v0,Quaternion v1) error: v1.w can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		v0.w /= v1.w;
		return v0;
	}

	__duel__ const Quaternion<Int>& operator/=(Quaternion<Int>& v, const Float& n) = delete;
	__duel__ const Quaternion<Int>& operator/=(Quaternion<Int>& v0, const Quaternion<Float>& v1) = delete;


#pragma endregion


#pragma endregion

#pragma region Quaternion different type operation

#pragma region Quaternion different type operation +

	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator+(const T& n, const Quaternion<U>& v)
	{
		return Quaternion<Float>(n + v.x, n + v.y, n + v.z, n + v.w);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator+(const Quaternion<T>& v, const U& n)
	{
		return Quaternion<Float>(v.x + n, v.y + n, v.z + n, v.w + n);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator+(const Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		return Quaternion<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
	}

#pragma endregion

#pragma region Quaternion different type operation -

	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator-(const T& n, const Quaternion<U>& v)
	{
		return Quaternion<Float>(n - v.x, n - v.y, n - v.z, n - v.w);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator-(const Quaternion<T>& v, const U& n)
	{
		return Quaternion<Float>(v.x - n, v.y - n, v.z - n, v.w - n);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator-(const Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		return Quaternion<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
	}

#pragma endregion

#pragma region Quaternion different type operation *

	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator*(const T& n, const Quaternion<U>& v)
	{
		return Quaternion<Float>(n * v.x, n * v.y, n * v.z, n * v.w);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator*(const Quaternion<T>& v, const U& n)
	{
		return Quaternion<Float>(v.x * n, v.y * n, v.z * n, v.w * n);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator*(const Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		Float a = v0.w;
		Float b = v1.w;
		Vec3<Float> u(v0.x, v0.y, v0.z);
		Vec3<Float> v(v1.x, v1.y, v1.z);
		auto ur = a * v + b * u + cross(u, v);
		Float w = a * b - dot(u, v);
		return Quaternion<Float>(ur, w);
	}

#pragma endregion

#pragma region Quaternion different type operation /

	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator/(const T& n, const Quaternion<U>& v)
	{
		CHECK(v.x != 0, "Quaternion<Float> operation /(n, Quaternion v1): v1.x can not be zero.");
		CHECK(v.y != 0, "Quaternion<Float> operation /(n, Quaternion v1): v1.y can not be zero.");
		CHECK(v.z != 0, "Quaternion<Float> operation /(n, Quaternion v1): v1.z can not be zero.");
		CHECK(v.w != 0, "Quaternion<Float> operation /(n, Quaternion v1): v1.w can not be zero.");
		return Quaternion<Float>(n / v.x, n / v.y, n / v.z, n / v.w);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator/(const Quaternion<T>& v, const U& n)
	{
		CHECK(n != 0, "Quaternion<Float> operation /(Quaternion v, n): n can not be zero.");
		return Quaternion<Float>(v.x / n, v.y / n, v.z / n, v.w / n);
	}
	template<typename T, typename U>
	__duel__ const Quaternion<Float> operator/(const Quaternion<T>& v0, const Quaternion<U>& v1)
	{
		CHECK(v1.x != 0, "Quaternion<Float> operation /(Quaternion v0, Quaternion v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "Quaternion<Float> operation /(Quaternion v0, Quaternion v1): v1.y can not be zero.");
		CHECK(v1.z != 0, "Quaternion<Float> operation /(Quaternion v0, Quaternion v1): v1.z can not be zero.");
		CHECK(v1.w != 0, "Quaternion<Float> operation /(Quaternion v0, Quaternion v1): v1.w can not be zero.");
		return Quaternion<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
	}

#pragma endregion


#pragma endregion

#pragma region Quaternion operation

	template<typename T>
	__duel__ const Float norm(const Quaternion<T>& qua)
	{
		Float square = qua.x*qua.x + qua.y*qua.y + qua.z*qua.z + qua.w*qua.w;
		return sqrt(square);
	}

	template<typename T>
	__duel__ const Quaternion<Float> normalize(const Quaternion<T>& vec)
	{
		Float square = vec.x*vec.x + vec.y*vec.y + vec.z*vec.z + vec.w*vec.w;
		CHECK(square > 0.0, "Quaternion normalize error: square can not less than 0.0!");
		Float norm = sqrt(square);
		Float inv = 1.0 / norm;
		return inv * vec;
	}

	template<typename T>
	__duel__ const Quaternion<T> conjugate(const Quaternion<T>& qua)
	{
		return Quaternion<T>(-qua.x, -qua.y, -qua.z, qua.w);
	}

	template<typename T>
	__duel__ const Quaternion<Float> inverse(const Quaternion<T>& qua)
	{
		Quaternion<Float> qConj = conjugate(qua);
		Float nor = norm(qua);
		Float nor2 = nor * nor;
		CHECK(nor2 != 0.0, "Quaternion inverse error: nor2 can not equal to 0!");
		Float inv = 1.0 / nor2;
		return inv * qConj;
	}

#pragma endregion

	template<typename T>
	__duel__ void logData(const Quaternion<T>& v)
	{
		const custd::OStream os;
		os << v.x << "\t" << v.y << "\t" << v.z << "\t" << v.w << custd::endl;
	}

#pragma endregion

#pragma region Color3
	template<typename T>
	class Color3
	{
	public:
		T r, g, b;
	public:
		__duel__ Color3() :r(0), g(0), b(0) {}
		__duel__ Color3(const T& _x, const T& _y, const T& _z) : r(_x), g(_y), b(_z) {}
		__duel__ Color3(const T& n) : r(n), g(n), b(n) {}
		__duel__ Color3(const Color3<T>& v) : r(v.r), g(v.g), b(v.b) {}
		template<typename U>
		__duel__ explicit Color3(const Color3<U>& v) : r(v.r), g(v.g), b(v.b) {}
		__duel__ ~Color3() {}
	public:
		__duel__ const Color3& operator=(const Color3<int>& v)
		{
			r = v.r;
			g = v.g;
			b = v.b;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "The <idx> in Color3<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return r; break;
			case 1: return g; break;
			case 2: return b; break;
			default: CHECK(false, "Can not run Color3::operator[idx]: switch::default."); break;
			}
		}
	};

	typedef Color3<Int> Color3i;
	typedef Color3<Float> Color3f;

#pragma region Color process function

	template<typename T>
	__duel__ const Color3<Float> calculateGammaColor(const Color3<T>& color, const Float& gamma)
	{
		CHECK(gamma >= 1.0&&gamma <= 100.0, "The input gamma is out of range!");
		Float power = 1.0 / gamma;
		return Color3<Float>(pow(color.r, power), pow(color.g, power), pow(color.b, power));
	}
	template<typename T>
	__duel__ const Color3<Int> calculateGradeColor(const Color3<T>& color, const Float& maxVal)
	{
		CHECK(maxVal > 1.0, "The input maxVal can not less than 1.0!");
		const Float epcilon = 0.5f;
		Float inv = 1.0 / maxVal;
		Float rf = inv * color.r;
		Float gf = inv * color.g;
		Float bf = inv * color.b;
		Int ri = Int(round(r) + epcilon);
		Int gi = Int(round(g) + epcilon);
		Int bi = Int(round(b) + epcilon);
		return Color3<Int>(ri, gi, bi);
	}

#pragma endregion

#pragma region Color3 same type operation

#pragma region Color3 same type operation +

	template<typename T>
	__duel__ const Color3<T> operator+(const T& n, const Color3<T>& v)
	{
		return Color3<T>(n + v.r, n + v.g, n + v.b);
	}
	template<typename T>
	__duel__ const Color3<T> operator+(const Color3<T>& v, const T& n)
	{
		return Color3<T>(v.r + n, v.g + n, v.b + n);
	}
	template<typename T>
	__duel__ const Color3<T> operator+(const Color3<T>& v0, const Color3<T>& v1)
	{
		return Color3<T>(v0.r + v1.r, v0.g + v1.g, v0.b + v1.b);
	}

	template<typename T, typename U>
	__duel__ const Color3<T>& operator+=(Color3<T>& v, const U& n)
	{
		v.r += n;
		v.g += n;
		v.b += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Color3<T>& operator+=(Color3<T>& v0, const Color3<U>& v1)
	{
		v0.r += v1.r;
		v0.g += v1.g;
		v0.b += v1.b;
		return v0;
	}

	__duel__ const Color3<Int>& operator+=(Color3<Int>& v, const Float& n) = delete;
	__duel__ const Color3<Int>& operator+=(Color3<Int>& v0, const Color3<Float>& v1) = delete;


#pragma endregion

#pragma region Color3 same type operation -

	template<typename T>
	__duel__ const Color3<T> operator-(const T& n, const Color3<T>& v)
	{
		return Color3<T>(n - v.r, n - v.g, n - v.b);
	}
	template<typename T>
	__duel__ const Color3<T> operator-(const Color3<T>& v, const T& n)
	{
		return Color3<T>(v.r - n, v.g - n, v.b - n);
	}
	template<typename T>
	__duel__ const Color3<T> operator-(const Color3<T>& v0, const Color3<T>& v1)
	{
		return Color3<T>(v0.r - v1.r, v0.g - v1.g, v0.b - v1.b);
	}

	template<typename T, typename U>
	__duel__ const Color3<T>& operator-=(Color3<T>& v, const U& n)
	{
		v.r -= n;
		v.g -= n;
		v.b -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Color3<T>& operator-=(Color3<T>& v0, const Color3<U>& v1)
	{
		v0.r -= v1.r;
		v0.g -= v1.g;
		v0.b -= v1.b;
		return v0;
	}

	__duel__ const Color3<Int>& operator-=(Color3<Int>& v, const Float& n) = delete;
	__duel__ const Color3<Int>& operator-=(Color3<Int>& v0, const Color3<Float>& v1) = delete;


#pragma endregion

#pragma region Color3 same type operation *

	template<typename T>
	__duel__ const Color3<T> operator*(const T& n, const Color3<T>& v)
	{
		return Color3<T>(n * v.r, n * v.g, n * v.b);
	}
	template<typename T>
	__duel__ const Color3<T> operator*(const Color3<T>& v, const T& n)
	{
		return Color3<T>(v.r * n, v.g * n, v.b * n);
	}
	template<typename T>
	__duel__ const Color3<T> operator*(const Color3<T>& v0, const Color3<T>& v1)
	{
		return Color3<T>(v0.r * v1.r, v0.g * v1.g, v0.b * v1.b);
	}

	template<typename T, typename U>
	__duel__ const Color3<T>& operator*=(Color3<T>& v, const U& n)
	{
		v.r *= n;
		v.g *= n;
		v.b *= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Color3<T>& operator*=(Color3<T>& v0, const Color3<U>& v1)
	{
		v0.r *= v1.r;
		v0.g *= v1.g;
		v0.b *= v1.b;
		return v0;
	}

	__duel__ const Color3<Int>& operator*=(Color3<Int>& v, const Float& n) = delete;
	__duel__ const Color3<Int>& operator*=(Color3<Int>& v0, const Color3<Float>& v1) = delete;


#pragma endregion

#pragma region Color3 same type operation /

	template<typename T>
	__duel__ const Color3<T> operator/(const T& n, const Color3<T>& v)
	{
		CHECK(v.r != 0, "Same type Color3 operator/(n,Color3 v) error: v.x can not be 0!");
		CHECK(v.g != 0, "Same type Color3 operator/(n,Color3 v) error: v.y can not be 0!");
		CHECK(v.b != 0, "Same type Color3 operator/(n,Color3 v) error: v.z can not be 0!");
		return Color3<T>(n / v.r, n / v.g, n / v.b);
	}
	template<typename T>
	__duel__ const Color3<T> operator/(const Color3<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type Color3 operator/(Color3 v, n) error: n can not be 0!");
		return Color3<T>(v.r / n, v.g / n, v.b / n);
	}
	template<typename T>
	__duel__ const Color3<T> operator/(const Color3<T>& v0, const Color3<T>& v1)
	{
		CHECK(v1.r != 0, "Same type Color3 operator/(n,Color3 v) error: v1.x can not be 0!");
		CHECK(v1.g != 0, "Same type Color3 operator/(n,Color3 v) error: v1.y can not be 0!");
		CHECK(v1.b != 0, "Same type Color3 operator/(n,Color3 v) error: v1.z can not be 0!");
		return Color3<T>(v0.r / v1.r, v0.g / v1.g, v0.b / v1.b);
	}

	template<typename T, typename U>
	__duel__ const Color3<T>& operator/=(Color3<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type Color3 operator/=(Color3 v, n) error: n can not be 0!");
		v.r /= n;
		v.g /= n;
		v.b /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const Color3<T>& operator/=(Color3<T>& v0, const Color3<U>& v1)
	{
		CHECK(v1.r != 0, "Same type Color3 operator/=(Color3 v0,Color3 v1) error: v1.x can not be 0!");
		CHECK(v1.g != 0, "Same type Color3 operator/=(Color3 v0,Color3 v1) error: v1.y can not be 0!");
		CHECK(v1.b != 0, "Same type Color3 operator/=(Color3 v0,Color3 v1) error: v1.z can not be 0!");
		v0.r /= v1.r;
		v0.g /= v1.g;
		v0.b /= v1.b;
		return v0;
	}

	__duel__ const Color3<Int>& operator/=(Color3<Int>& v, const Float& n) = delete;
	__duel__ const Color3<Int>& operator/=(Color3<Int>& v0, const Color3<Float>& v1) = delete;


#pragma endregion

#pragma endregion

#pragma region Color3 different type operation

#pragma region Color3 different type operation +

	template<typename T, typename U>
	__duel__ const Color3<Float> operator+(const T& n, const Color3<U>& v)
	{
		return Color3<Float>(n + v.r, n + v.g);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator+(const Color3<T>& v, const U& n)
	{
		return Color3<Float>(v.r + n, v.g + n);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator+(const Color3<T>& v0, const Color3<U>& v1)
	{
		return Color3<Float>(v0.r + v1.r, v0.g + v1.g);
	}

#pragma endregion

#pragma region Color3 different type operation -

	template<typename T, typename U>
	__duel__ const Color3<Float> operator-(const T& n, const Color3<U>& v)
	{
		return Color3<Float>(n - v.r, n - v.g);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator-(const Color3<T>& v, const U& n)
	{
		return Color3<Float>(v.r - n, v.g - n);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator-(const Color3<T>& v0, const Color3<U>& v1)
	{
		return Color3<Float>(v0.r - v1.r, v0.g - v1.g);
	}

#pragma endregion

#pragma region Color3 different type operation *

	template<typename T, typename U>
	__duel__ const Color3<Float> operator*(const T& n, const Color3<U>& v)
	{
		return Color3<Float>(n * v.r, n * v.g);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator*(const Color3<T>& v, const U& n)
	{
		return Color3<Float>(v.r * n, v.g * n);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator*(const Color3<T>& v0, const Color3<U>& v1)
	{
		return Color3<Float>(v0.r * v1.r, v0.g * v1.g);
	}

#pragma endregion

#pragma region Color3 different type operation /

	template<typename T, typename U>
	__duel__ const Color3<Float> operator/(const T& n, const Color3<U>& v)
	{
		CHECK(v.r != 0, "Color3<Float> operation /(n, Color3 v1): v1.x can not be zero.");
		CHECK(v.g != 0, "Color3<Float> operation /(n, Color3 v1): v1.y can not be zero.");
		CHECK(v.b != 0, "Color3<Float> operation /(n, Color3 v1): v1.z can not be zero.");
		return Color3<Float>(n / v.r, n / v.g);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator/(const Color3<T>& v, const U& n)
	{
		CHECK(n != 0, "Color3<Float> operation /(Color3 v, n): n can not be zero.");
		return Color3<Float>(v.r / n, v.g / n);
	}
	template<typename T, typename U>
	__duel__ const Color3<Float> operator/(const Color3<T>& v0, const Color3<U>& v1)
	{
		CHECK(v1.r != 0, "Color3<Float> operation /(Color3 v0, Color3 v1): v1.x can not be zero.");
		CHECK(v1.g != 0, "Color3<Float> operation /(Color3 v0, Color3 v1): v1.y can not be zero.");
		CHECK(v1.b != 0, "Color3<Float> operation /(Color3 v0, Color3 v1): v1.z can not be zero.");
		return Color3<Float>(v0.r / v1.r, v0.g / v1.g);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const Color3<T>& v)
	{
		const custd::OStream os;
		os << v.r << "\t" << v.g << "\t" << v.b << custd::endl;
	}

#pragma endregion

#pragma region Mat3x3
	template<typename T>
	class Mat3x3
	{
	public:
		T m[3][3];
	public:
		__duel__ Mat3x3()
		{
			for (Int i = 0; i < 3; i++)
			{
				for (Int j = 0; j < 3; j++)
				{
					m[i][j] = 0;
				}
			}
		}
		__duel__ Mat3x3(
			const T& m00, const T& m01, const T& m02,
			const T& m10, const T& m11, const T& m12,
			const T& m20, const T& m21, const T& m22
		)
		{
			m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
			m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
			m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
		}
		__duel__ Mat3x3(const T& n)
		{
			for (Int i = 0; i < 3; i++)
			{
				for (Int j = 0; j < 3; j++)
				{
					m[i][j] = n;
				}
			}
		}
		__duel__ Mat3x3(const Mat3x3<T>& mat)
		{
			for (Int i = 0; i < 3; i++)
			{
				for (Int j = 0; j < 3; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		template<typename U>
		__duel__ explicit Mat3x3(const Mat3x3<U>& mat)
		{
			for (Int i = 0; i < 3; i++)
			{
				for (Int j = 0; j < 3; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		__duel__ ~Mat3x3()
		{

		}
	public:

		Mat3x3(const Vec3<T>& v0, const Vec3<T>& v1, const Vec3<T>& v2, const Bool& isColumn = false)
		{
			if (isColumn)
			{
				for (Int i = 0; i < 3; i++)
				{
					m[i][0] = v0[i];
					m[i][1] = v1[i];
					m[i][2] = v2[i];
				}
			}
			else
			{
				for (Int i = 0; i < 3; i++)
				{
					m[0][i] = v0[i];
					m[1][i] = v1[i];
					m[2][i] = v2[i];
				}
			}
		}

	public:
		__duel__ const Mat3x3& operator=(const Mat3x3<Int>& mat)
		{
			for (Int i = 0; i < 3; i++)
			{
				for (Int j = 0; j < 3; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
	public:
		const Vec3<T> GetRow(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "Mat3x3::GetRow(idx) error: idx is out of range!");
			return Vec3<T>(m[idx][0], m[idx][1], m[idx][2]);
		}
		const Vec3<T> GetColumn(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "Mat3x3::GetColumn(idx) error: idx is out of range!");
			return Vec3<T>(m[0][idx], m[1][idx], m[2][idx]);
		}
		const Vec3<T> GetDiag()
		{
			return Vec3<T>(m[0][0], m[1][1], m[2][2]);
		}
	};

	typedef Mat3x3<Int> Mat3x3i;
	typedef Mat3x3<Float> Mat3x3f;

#pragma region Mat3x3 same type operation

#pragma region Mat3x3 same type operation +
	template<typename T>
	__duel__ const Mat3x3<T> operator+(const T&n, const Mat3x3<T>& mat)
	{
		return Mat3x3<T>(
			n + mat.m[0][0], n + mat.m[0][1], n + mat.m[0][2],
			n + mat.m[1][0], n + mat.m[1][1], n + mat.m[1][2],
			n + mat.m[2][0], n + mat.m[2][1], n + mat.m[2][2]);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator+(const Mat3x3<T>& mat, const T&n)
	{
		return Mat3x3<T>(
			mat.m[0][0] + n, mat.m[0][1] + n, mat.m[0][2] + n,
			mat.m[1][0] + n, mat.m[1][1] + n, mat.m[1][2] + n,
			mat.m[2][0] + n, mat.m[2][1] + n, mat.m[2][2] + n);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator+(const Mat3x3<T>& mat0, const Mat3x3<T>& mat1)
	{
		return Mat3x3<T>(
			mat0.m[0][0] + mat1.m[0][0], mat0.m[0][1] + mat1.m[0][1], mat0.m[0][2] + mat1.m[0][2],
			mat0.m[1][0] + mat1.m[1][0], mat0.m[1][1] + mat1.m[1][1], mat0.m[1][2] + mat1.m[1][2],
			mat0.m[2][0] + mat1.m[2][0], mat0.m[2][1] + mat1.m[2][1], mat0.m[2][2] + mat1.m[2][2]);
	}

	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator+=(Mat3x3<T>& mat, const U& n)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat.m[i][j] += n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator+=(Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat0.m[i][j] += mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat3x3<Int>& operator+=(Mat3x3<Int>& mat, const Float& n) = delete;
	__duel__ const Mat3x3<Int>& operator+=(Mat3x3<Int>& mat0, const Mat3x3<Float>& mat1) = delete;

#pragma endregion

#pragma region Mat3x3 same type operation -
	template<typename T>
	__duel__ const Mat3x3<T> operator-(const T&n, const Mat3x3<T>& mat)
	{
		return Mat3x3<T>(
			n - mat.m[0][0], n - mat.m[0][1], n - mat.m[0][2],
			n - mat.m[1][0], n - mat.m[1][1], n - mat.m[1][2],
			n - mat.m[2][0], n - mat.m[2][1], n - mat.m[2][2]);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator-(const Mat3x3<T>& mat, const T&n)
	{
		return Mat3x3<T>(
			mat.m[0][0] - n, mat.m[0][1] - n, mat.m[0][2] - n,
			mat.m[1][0] - n, mat.m[1][1] - n, mat.m[1][2] - n,
			mat.m[2][0] - n, mat.m[2][1] - n, mat.m[2][2] - n);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator-(const Mat3x3<T>& mat0, const Mat3x3<T>& mat1)
	{
		return Mat3x3<T>(
			mat0.m[0][0] - mat1.m[0][0], mat0.m[0][1] - mat1.m[0][1], mat0.m[0][2] - mat1.m[0][2],
			mat0.m[1][0] - mat1.m[1][0], mat0.m[1][1] - mat1.m[1][1], mat0.m[1][2] - mat1.m[1][2],
			mat0.m[2][0] - mat1.m[2][0], mat0.m[2][1] - mat1.m[2][1], mat0.m[2][2] - mat1.m[2][2]);
	}

	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator-=(Mat3x3<T>& mat, const U& n)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat.m[i][j] -= n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator-=(Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat0.m[i][j] -= mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat3x3<Int>& operator-=(Mat3x3<Int>& mat, const Float& n) = delete;
	__duel__ const Mat3x3<Int>& operator-=(Mat3x3<Int>& mat0, const Mat3x3<Float>& mat1) = delete;

#pragma endregion

#pragma region Mat3x3 same type operation *
	template<typename T>
	__duel__ const Mat3x3<T> operator*(const T&n, const Mat3x3<T>& mat)
	{
		return Mat3x3<T>(
			n * mat.m[0][0], n * mat.m[0][1], n * mat.m[0][2],
			n * mat.m[1][0], n * mat.m[1][1], n * mat.m[1][2],
			n * mat.m[2][0], n * mat.m[2][1], n * mat.m[2][2]);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator*(const Mat3x3<T>& mat, const T&n)
	{
		return Mat3x3<T>(
			mat.m[0][0] * n, mat.m[0][1] * n, mat.m[0][2] * n,
			mat.m[1][0] * n, mat.m[1][1] * n, mat.m[1][2] * n,
			mat.m[2][0] * n, mat.m[2][1] * n, mat.m[2][2] * n);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator*(const Mat3x3<T>& mat0, const Mat3x3<T>& mat1)
	{
		Mat3x3<T> result;
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				result.m[i][j] = dot(mat0.GetRow(i), mat1.GetColumn(j));
			}
		}
		return result;
	}

	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator*=(Mat3x3<T>& mat, const U& n)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat.m[i][j] *= n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator*=(Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat0.m[i][j] *= mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat3x3<Int>& operator*=(Mat3x3<Int>& mat, const Float& n) = delete;
	__duel__ const Mat3x3<Int>& operator*=(Mat3x3<Int>& mat0, const Mat3x3<Float>& mat1) = delete;

#pragma endregion

#pragma region Mat3x3 same type operation /
	template<typename T>
	__duel__ const Mat3x3<T> operator/(const T&n, const Mat3x3<T>& mat)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				CHECK(mat.m[i][j] != 0, "Same type Mat3x3 operation /(n, Mat3x3 mat) error: one component of the mat can not be 0!");
			}
		}
		return Mat3x3<T>(
			n / mat.m[0][0], n / mat.m[0][1], n / mat.m[0][2],
			n / mat.m[1][0], n / mat.m[1][1], n / mat.m[1][2],
			n / mat.m[2][0], n / mat.m[2][1], n / mat.m[2][2]);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator/(const Mat3x3<T>& mat, const T&n)
	{
		CHECK(n != 0, "Same type Mat3x3 operation /(Mat3x3 mat, n) error: n can not be 0!");
		return Mat3x3<T>(
			mat.m[0][0] / n, mat.m[0][1] / n, mat.m[0][2] / n,
			mat.m[1][0] / n, mat.m[1][1] / n, mat.m[1][2] / n,
			mat.m[2][0] / n, mat.m[2][1] / n, mat.m[2][2] / n);
	}
	template<typename T>
	__duel__ const Mat3x3<T> operator/(const Mat3x3<T>& mat0, const Mat3x3<T>& mat1)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				CHECK(mat1.m[i][j] != 0, "Same type Mat3x3 operation /(n, Mat3x3 mat) error: one component of the mat1 can not be 0!");
			}
		}
		return Mat3x3<T>(
			mat0.m[0][0] / mat1.m[0][0], mat0.m[0][1] / mat1.m[0][1], mat0.m[0][2] / mat1.m[0][2],
			mat0.m[1][0] / mat1.m[1][0], mat0.m[1][1] / mat1.m[1][1], mat0.m[1][2] / mat1.m[1][2],
			mat0.m[2][0] / mat1.m[2][0], mat0.m[2][1] / mat1.m[2][1], mat0.m[2][2] / mat1.m[2][2]);
	}

	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator/=(Mat3x3<T>& mat, const U& n)
	{
		CHECK(n != 0, "Same type Mat3x3 operation /=(Mat3x3 mat, n) error: n can not be 0!");
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat.m[i][j] /= n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<T>& operator/=(Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				CHECK(mat1.m[i][j] != 0, "Same type Mat3x3 operation /=(n, Mat3x3 mat) error: one component of the mat1 can not be 0!");
			}
		}
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				mat0.m[i][j] /= mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat3x3<Int>& operator/=(Mat3x3<Int>& mat, const Float& n) = delete;
	__duel__ const Mat3x3<Int>& operator/=(Mat3x3<Int>& mat0, const Mat3x3<Float>& mat1) = delete;

#pragma endregion

#pragma endregion

#pragma region Mat3x3 different type operation

#pragma region Mat3x3 different type operation +
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator+(const T&n, const Mat3x3<U>& mat)
	{
		return Mat3x3<Float>(
			n + mat.m[0][0], n + mat.m[0][1], n + mat.m[0][2],
			n + mat.m[1][0], n + mat.m[1][1], n + mat.m[1][2],
			n + mat.m[2][0], n + mat.m[2][1], n + mat.m[2][2]);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator+(const Mat3x3<U>& mat, const T&n)
	{
		return Mat3x3<Float>(
			mat.m[0][0] + n, mat.m[0][1] + n, mat.m[0][2] + n,
			mat.m[1][0] + n, mat.m[1][1] + n, mat.m[1][2] + n,
			mat.m[2][0] + n, mat.m[2][1] + n, mat.m[2][2] + n);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator+(const Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		return Mat3x3<Float>(
			mat0.m[0][0] + mat1.m[0][0], mat0.m[0][1] + mat1.m[0][1], mat0.m[0][2] + mat1.m[0][2],
			mat0.m[1][0] + mat1.m[1][0], mat0.m[1][1] + mat1.m[1][1], mat0.m[1][2] + mat1.m[1][2],
			mat0.m[2][0] + mat1.m[2][0], mat0.m[2][1] + mat1.m[2][1], mat0.m[2][2] + mat1.m[2][2]);
	}
#pragma endregion

#pragma region Mat3x3 different type operation -
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator-(const T&n, const Mat3x3<U>& mat)
	{
		return Mat3x3<Float>(
			n - mat.m[0][0], n - mat.m[0][1], n - mat.m[0][2],
			n - mat.m[1][0], n - mat.m[1][1], n - mat.m[1][2],
			n - mat.m[2][0], n - mat.m[2][1], n - mat.m[2][2]);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator-(const Mat3x3<U>& mat, const T&n)
	{
		return Mat3x3<Float>(
			mat.m[0][0] - n, mat.m[0][1] - n, mat.m[0][2] - n,
			mat.m[1][0] - n, mat.m[1][1] - n, mat.m[1][2] - n,
			mat.m[2][0] - n, mat.m[2][1] - n, mat.m[2][2] - n);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator-(const Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		return Mat3x3<Float>(
			mat0.m[0][0] - mat1.m[0][0], mat0.m[0][1] - mat1.m[0][1], mat0.m[0][2] - mat1.m[0][2],
			mat0.m[1][0] - mat1.m[1][0], mat0.m[1][1] - mat1.m[1][1], mat0.m[1][2] - mat1.m[1][2],
			mat0.m[2][0] - mat1.m[2][0], mat0.m[2][1] - mat1.m[2][1], mat0.m[2][2] - mat1.m[2][2]);
	}
#pragma endregion

#pragma region Mat3x3 different type operation *
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator*(const T&n, const Mat3x3<U>& mat)
	{
		return Mat3x3<Float>(
			n * mat.m[0][0], n * mat.m[0][1], n * mat.m[0][2],
			n * mat.m[1][0], n * mat.m[1][1], n * mat.m[1][2],
			n * mat.m[2][0], n * mat.m[2][1], n * mat.m[2][2]);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator*(const Mat3x3<U>& mat, const T&n)
	{
		return Mat3x3<Float>(
			mat.m[0][0] * n, mat.m[0][1] * n, mat.m[0][2] * n,
			mat.m[1][0] * n, mat.m[1][1] * n, mat.m[1][2] * n,
			mat.m[2][0] * n, mat.m[2][1] * n, mat.m[2][2] * n);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator*(const Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		Mat3x3<Float> result;
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				result.m[i][j] = dot(mat0.GetRow(i), mat1.GetColumn(j));
			}
		}
		return result;
	}
#pragma endregion

#pragma region Mat3x3 different type operation /
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator/(const T&n, const Mat3x3<U>& mat)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				CHECK(mat.m[i][j] != 0, "Different type Mat3x3 operation /(n, Mat3x3 mat) error: one component of the mat can not be 0!");
			}
		}
		return Mat3x3<Float>(
			n / mat.m[0][0], n / mat.m[0][1], n / mat.m[0][2],
			n / mat.m[1][0], n / mat.m[1][1], n / mat.m[1][2],
			n / mat.m[2][0], n / mat.m[2][1], n / mat.m[2][2]);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator/(const Mat3x3<U>& mat, const T&n)
	{
		CHECK(n != 0, "Different type Mat3x3 operation /=(Mat3x3 mat, n) error: n can not be 0!");
		return Mat3x3<Float>(
			mat.m[0][0] / n, mat.m[0][1] / n, mat.m[0][2] / n,
			mat.m[1][0] / n, mat.m[1][1] / n, mat.m[1][2] / n,
			mat.m[2][0] / n, mat.m[2][1] / n, mat.m[2][2] / n);
	}
	template<typename T, typename U>
	__duel__ const Mat3x3<Float> operator/(const Mat3x3<T>& mat0, const Mat3x3<U>& mat1)
	{
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				CHECK(mat1.m[i][j] != 0, "Different type Mat3x3 operation /(Mat3x3 mat0, Mat3x3 mat1) error: one component of the mat can not be 0!");
			}
		}
		return Mat3x3<Float>(
			mat0.m[0][0] / mat1.m[0][0], mat0.m[0][1] / mat1.m[0][1], mat0.m[0][2] / mat1.m[0][2],
			mat0.m[1][0] / mat1.m[1][0], mat0.m[1][1] / mat1.m[1][1], mat0.m[1][2] / mat1.m[1][2],
			mat0.m[2][0] / mat1.m[2][0], mat0.m[2][1] / mat1.m[2][1], mat0.m[2][2] / mat1.m[2][2]);
	}
#pragma endregion

#pragma endregion

#pragma region Mat3x3 marco
#define Mat3x3_identity Mat3x3<Int>(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
	//#define Mat3x3_identity Mat3x3(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
#pragma endregion

	template<typename T>
	__duel__ void logData(const Mat3x3<T>& mat)
	{
		const custd::OStream os;
		for (Int i = 0; i < 3; i++)
		{
			for (Int j = 0; j < 3; j++)
			{
				os << mat.m[i][j] << "\t";
			}
			os << custd::endl;
		}
	}

#pragma endregion

#pragma region Mat4x4
	template<typename T>
	class Mat4x4
	{
	public:
		T m[4][4];
	public:
		__duel__ Mat4x4()
		{
			for (Int i = 0; i < 4; i++)
			{
				for (Int j = 0; j < 4; j++)
				{
					m[i][j] = 0;
				}
			}
		}
		__duel__ Mat4x4(
			const T& m00, const T& m01, const T& m02, const T& m03,
			const T& m10, const T& m11, const T& m12, const T& m13,
			const T& m20, const T& m21, const T& m22, const T& m23,
			const T& m30, const T& m31, const T& m32, const T& m33
		)
		{
			m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
			m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
			m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
			m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
		}
		__duel__ Mat4x4(const T& n)
		{
			for (Int i = 0; i < 4; i++)
			{
				for (Int j= 0; j < 4; j++)
				{
					m[i][j] = n;
				}
			}
		}
		__duel__ Mat4x4(const Mat4x4<T>& mat)
		{
			for (Int i = 0; i < 4; i++)
			{
				for (Int j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		template<typename U>
		__duel__ explicit Mat4x4(const Mat4x4<U>& mat)
		{
			for (Int i = 0; i < 4; i++)
			{
				for (Int j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		__duel__ ~Mat4x4()
		{

		}
	public:
		Mat4x4(const Vec4<T>& v0, const Vec4<T>& v1, const Vec4<T>& v2, const Vec4<T>& v3, const Bool& isColumn = false)
		{
			if (isColumn)
			{
				for (Int i = 0; i < 4; i++)
				{
					m[i][0] = v0[i];
					m[i][1] = v1[i];
					m[i][2] = v2[i];
					m[i][3] = v3[i];
				}
			}
			else
			{
				for (Int i = 0; i < 4; i++)
				{
					m[0][i] = v0[i];
					m[1][i] = v1[i];
					m[2][i] = v2[i];
					m[3][i] = v3[i];
				}
			}
		}
	public:
		__duel__ const Mat4x4& operator=(const Mat4x4<Int>& mat)
		{
			for (Int i = 0; i < 4; i++)
			{
				for (Int j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
	public:
		const Vec4<T> GetRow(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 3, "Mat4x4::GetRow(idx) error: idx is out of range!");
			return Vec4<T>(m[idx][0], m[idx][1], m[idx][2], m[idx][3]);
		}
		const Vec4<T> GetColumn(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 3, "Mat4x4::GetColumn(idx) error: idx is out of range!");
			return Vec4<T>(m[0][idx], m[1][idx], m[2][idx], m[3][idx]);
		}
		const Vec4<T> GetDiag()
		{
			return Vec4<T>(m[0][0], m[1][1], m[2][2], m[3][3]);
		}
	};

	typedef Mat4x4<Int> Mat4x4i;
	typedef Mat4x4<Float> Mat4x4f;

#pragma region Mat4x4 same type operation

#pragma region Mat4x4 same type operation +
	template<typename T>
	__duel__ const Mat4x4<T> operator+(const T&n, const Mat4x4<T>& mat)
	{
		return Mat4x4<T>(
			n + mat.m[0][0], n + mat.m[0][1], n + mat.m[0][2], n + mat.m[0][3],
			n + mat.m[1][0], n + mat.m[1][1], n + mat.m[1][2], n + mat.m[1][3],
			n + mat.m[2][0], n + mat.m[2][1], n + mat.m[2][2], n + mat.m[2][3],
			n + mat.m[3][0], n + mat.m[3][1], n + mat.m[3][2], n + mat.m[3][3]);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator+(const Mat4x4<T>& mat, const T&n)
	{
		return Mat4x4<T>(
			mat.m[0][0] + n, mat.m[0][1] + n, mat.m[0][2] + n, mat.m[0][3] + n,
			mat.m[1][0] + n, mat.m[1][1] + n, mat.m[1][2] + n, mat.m[1][3] + n,
			mat.m[2][0] + n, mat.m[2][1] + n, mat.m[2][2] + n, mat.m[2][3] + n,
			mat.m[3][0] + n, mat.m[3][1] + n, mat.m[3][2] + n, mat.m[3][3] + n);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator+(const Mat4x4<T>& mat0, const Mat4x4<T>& mat1)
	{
		return Mat4x4<T>(
			mat0.m[0][0] + mat1.m[0][0], mat0.m[0][1] + mat1.m[0][1], mat0.m[0][2] + mat1.m[0][2], mat0.m[0][3] + mat1.m[0][3],
			mat0.m[1][0] + mat1.m[1][0], mat0.m[1][1] + mat1.m[1][1], mat0.m[1][2] + mat1.m[1][2], mat0.m[1][3] + mat1.m[1][3],
			mat0.m[2][0] + mat1.m[2][0], mat0.m[2][1] + mat1.m[2][1], mat0.m[2][2] + mat1.m[2][2], mat0.m[2][3] + mat1.m[2][3],
			mat0.m[3][0] + mat1.m[3][0], mat0.m[3][1] + mat1.m[3][1], mat0.m[3][2] + mat1.m[3][2], mat0.m[3][3] + mat1.m[3][3]);
	}

	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator+=(Mat4x4<T>& mat, const U& n)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat.m[i][j] += n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator+=(Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat0.m[i][j] += mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat4x4<Int>& operator+=(Mat4x4<Int>& mat, const Float& n) = delete;
	__duel__ const Mat4x4<Int>& operator+=(Mat4x4<Int>& mat0, const Mat4x4<Float>& mat1) = delete;

	#pragma endregion

#pragma region Mat4x4 same type operation -
	template<typename T>
	__duel__ const Mat4x4<T> operator-(const T&n, const Mat4x4<T>& mat)
	{
		return Mat4x4<T>(
			n - mat.m[0][0], n - mat.m[0][1], n - mat.m[0][2], n - mat.m[0][3],
			n - mat.m[1][0], n - mat.m[1][1], n - mat.m[1][2], n - mat.m[1][3],
			n - mat.m[2][0], n - mat.m[2][1], n - mat.m[2][2], n - mat.m[2][3],
			n - mat.m[3][0], n - mat.m[3][1], n - mat.m[3][2], n - mat.m[3][3]);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator-(const Mat4x4<T>& mat, const T&n)
	{
		return Mat4x4<T>(
			mat.m[0][0] - n, mat.m[0][1] - n, mat.m[0][2] - n, mat.m[0][3] - n,
			mat.m[1][0] - n, mat.m[1][1] - n, mat.m[1][2] - n, mat.m[1][3] - n,
			mat.m[2][0] - n, mat.m[2][1] - n, mat.m[2][2] - n, mat.m[2][3] - n,
			mat.m[3][0] - n, mat.m[3][1] - n, mat.m[3][2] - n, mat.m[3][3] - n);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator-(const Mat4x4<T>& mat0, const Mat4x4<T>& mat1)
	{
		return Mat4x4<T>(
			mat0.m[0][0] - mat1.m[0][0], mat0.m[0][1] - mat1.m[0][1], mat0.m[0][2] - mat1.m[0][2], mat0.m[0][3] - mat1.m[0][3],
			mat0.m[1][0] - mat1.m[1][0], mat0.m[1][1] - mat1.m[1][1], mat0.m[1][2] - mat1.m[1][2], mat0.m[1][3] - mat1.m[1][3],
			mat0.m[2][0] - mat1.m[2][0], mat0.m[2][1] - mat1.m[2][1], mat0.m[2][2] - mat1.m[2][2], mat0.m[2][3] - mat1.m[2][3],
			mat0.m[3][0] - mat1.m[3][0], mat0.m[3][1] - mat1.m[3][1], mat0.m[3][2] - mat1.m[3][2], mat0.m[3][3] - mat1.m[3][3]);
	}

	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator-=(Mat4x4<T>& mat, const U& n)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat.m[i][j] -= n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator-=(Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat0.m[i][j] -= mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat4x4<Int>& operator-=(Mat4x4<Int>& mat, const Float& n) = delete;
	__duel__ const Mat4x4<Int>& operator-=(Mat4x4<Int>& mat0, const Mat4x4<Float>& mat1) = delete;

#pragma endregion

#pragma region Mat4x4 same type operation *
	template<typename T>
	__duel__ const Mat4x4<T> operator*(const T&n, const Mat4x4<T>& mat)
	{
		return Mat4x4<T>(
			n * mat.m[0][0], n * mat.m[0][1], n * mat.m[0][2], n * mat.m[0][3],
			n * mat.m[1][0], n * mat.m[1][1], n * mat.m[1][2], n * mat.m[1][3],
			n * mat.m[2][0], n * mat.m[2][1], n * mat.m[2][2], n * mat.m[2][3],
			n * mat.m[3][0], n * mat.m[3][1], n * mat.m[3][2], n * mat.m[3][3]);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator*(const Mat4x4<T>& mat, const T&n)
	{
		return Mat4x4<T>(
			mat.m[0][0] * n, mat.m[0][1] * n, mat.m[0][2] * n, mat.m[0][3] * n,
			mat.m[1][0] * n, mat.m[1][1] * n, mat.m[1][2] * n, mat.m[1][3] * n,
			mat.m[2][0] * n, mat.m[2][1] * n, mat.m[2][2] * n, mat.m[2][3] * n,
			mat.m[3][0] * n, mat.m[3][1] * n, mat.m[3][2] * n, mat.m[3][3] * n);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator*(const Mat4x4<T>& mat0, const Mat4x4<T>& mat1)
	{
		Mat4x4<T> result;
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				result.m[i][j] = dot(mat0.GetRow(i), mat1.GetColumn(j));
			}
		}
		return result;
	}

	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator*=(Mat4x4<T>& mat, const U& n)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat.m[i][j] *= n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator*=(Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat0.m[i][j] *= mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat4x4<Int>& operator*=(Mat4x4<Int>& mat, const Float& n) = delete;
	__duel__ const Mat4x4<Int>& operator*=(Mat4x4<Int>& mat0, const Mat4x4<Float>& mat1) = delete;

#pragma endregion

#pragma region Mat4x4 same type operation /
	template<typename T>
	__duel__ const Mat4x4<T> operator/(const T&n, const Mat4x4<T>& mat)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				CHECK(mat.m[i][j] != 0, "Same type Mat4x4 operation /(n, Mat4x4 mat) error: one component of the mat can not be 0!");
			}
		}
		return Mat4x4<T>(
			n / mat.m[0][0], n / mat.m[0][1], n / mat.m[0][2], n / mat.m[0][3],
			n / mat.m[1][0], n / mat.m[1][1], n / mat.m[1][2], n / mat.m[1][3],
			n / mat.m[2][0], n / mat.m[2][1], n / mat.m[2][2], n / mat.m[2][3],
			n / mat.m[3][0], n / mat.m[3][1], n / mat.m[3][2], n / mat.m[3][3]);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator/(const Mat4x4<T>& mat, const T&n)
	{
		CHECK(n != 0, "Same type Mat4x4 operation /(Mat4x4 mat, n) error: n can not be 0!");
		return Mat4x4<T>(
			mat.m[0][0] / n, mat.m[0][1] / n, mat.m[0][2] / n, mat.m[0][3] / n,
			mat.m[1][0] / n, mat.m[1][1] / n, mat.m[1][2] / n, mat.m[1][3] / n,
			mat.m[2][0] / n, mat.m[2][1] / n, mat.m[2][2] / n, mat.m[2][3] / n,
			mat.m[3][0] / n, mat.m[3][1] / n, mat.m[3][2] / n, mat.m[3][3] / n);
	}
	template<typename T>
	__duel__ const Mat4x4<T> operator/(const Mat4x4<T>& mat0, const Mat4x4<T>& mat1)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				CHECK(mat1.m[i][j] != 0, "Same type Mat4x4 operation /(n, Mat4x4 mat) error: one component of the mat1 can not be 0!");
			}
		}
		return Mat4x4<T>(
			mat0.m[0][0] / mat1.m[0][0], mat0.m[0][1] / mat1.m[0][1], mat0.m[0][2] / mat1.m[0][2], mat0.m[0][3] / mat1.m[0][3],
			mat0.m[1][0] / mat1.m[1][0], mat0.m[1][1] / mat1.m[1][1], mat0.m[1][2] / mat1.m[1][2], mat0.m[1][3] / mat1.m[1][3],
			mat0.m[2][0] / mat1.m[2][0], mat0.m[2][1] / mat1.m[2][1], mat0.m[2][2] / mat1.m[2][2], mat0.m[2][3] / mat1.m[2][3],
			mat0.m[3][0] / mat1.m[3][0], mat0.m[3][1] / mat1.m[3][1], mat0.m[3][2] / mat1.m[3][2], mat0.m[3][3] / mat1.m[3][3]);
	}

	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator/=(Mat4x4<T>& mat, const U& n)
	{
		CHECK(n != 0, "Same type Mat4x4 operation /=(Mat4x4 mat, n) error: n can not be 0!");
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat.m[i][j] /= n;
			}
		}
		return mat;
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<T>& operator/=(Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				CHECK(mat1.m[i][j] != 0, "Same type Mat4x4 operation /=(n, Mat4x4 mat) error: one component of the mat1 can not be 0!");
			}
		}
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				mat0.m[i][j] /= mat1.m[i][j];
			}
		}
		return mat0;
	}

	__duel__ const Mat4x4<Int>& operator/=(Mat4x4<Int>& mat, const Float& n) = delete;
	__duel__ const Mat4x4<Int>& operator/=(Mat4x4<Int>& mat0, const Mat4x4<Float>& mat1) = delete;

#pragma endregion


#pragma endregion

#pragma region Mat4x4 different type operation

#pragma region Mat4x4 different type operation +
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator+(const T&n, const Mat4x4<U>& mat)
	{
		return Mat4x4<Float>(
			n + mat.m[0][0], n + mat.m[0][1], n + mat.m[0][2], n + mat.m[0][3],
			n + mat.m[1][0], n + mat.m[1][1], n + mat.m[1][2], n + mat.m[1][3],
			n + mat.m[2][0], n + mat.m[2][1], n + mat.m[2][2], n + mat.m[2][3],
			n + mat.m[3][0], n + mat.m[3][1], n + mat.m[3][2], n + mat.m[3][3]);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator+(const Mat4x4<U>& mat, const T&n)
	{
		return Mat4x4<Float>(
			mat.m[0][0] + n, mat.m[0][1] + n, mat.m[0][2] + n, mat.m[0][3] + n,
			mat.m[1][0] + n, mat.m[1][1] + n, mat.m[1][2] + n, mat.m[1][3] + n,
			mat.m[2][0] + n, mat.m[2][1] + n, mat.m[2][2] + n, mat.m[2][3] + n,
			mat.m[3][0] + n, mat.m[3][1] + n, mat.m[3][2] + n, mat.m[3][3] + n);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator+(const Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		return Mat4x4<Float>(
			mat0.m[0][0] + mat1.m[0][0], mat0.m[0][1] + mat1.m[0][1], mat0.m[0][2] + mat1.m[0][2], mat0.m[0][3] + mat1.m[0][3],
			mat0.m[1][0] + mat1.m[1][0], mat0.m[1][1] + mat1.m[1][1], mat0.m[1][2] + mat1.m[1][2], mat0.m[1][3] + mat1.m[1][3],
			mat0.m[2][0] + mat1.m[2][0], mat0.m[2][1] + mat1.m[2][1], mat0.m[2][2] + mat1.m[2][2], mat0.m[2][3] + mat1.m[2][3],
			mat0.m[3][0] + mat1.m[3][0], mat0.m[3][1] + mat1.m[3][1], mat0.m[3][2] + mat1.m[3][2], mat0.m[3][3] + mat1.m[3][3]);
	}
#pragma endregion

#pragma region Mat4x4 different type operation -
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator-(const T&n, const Mat4x4<U>& mat)
	{
		return Mat4x4<Float>(
			n - mat.m[0][0], n - mat.m[0][1], n - mat.m[0][2], n - mat.m[0][3],
			n - mat.m[1][0], n - mat.m[1][1], n - mat.m[1][2], n - mat.m[1][3],
			n - mat.m[2][0], n - mat.m[2][1], n - mat.m[2][2], n - mat.m[2][3],
			n - mat.m[3][0], n - mat.m[3][1], n - mat.m[3][2], n - mat.m[3][3]);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator-(const Mat4x4<U>& mat, const T&n)
	{
		return Mat4x4<Float>(
			mat.m[0][0] - n, mat.m[0][1] - n, mat.m[0][2] - n, mat.m[0][3] - n,
			mat.m[1][0] - n, mat.m[1][1] - n, mat.m[1][2] - n, mat.m[1][3] - n,
			mat.m[2][0] - n, mat.m[2][1] - n, mat.m[2][2] - n, mat.m[2][3] - n,
			mat.m[3][0] - n, mat.m[3][1] - n, mat.m[3][2] - n, mat.m[3][3] - n);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator-(const Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		return Mat4x4<Float>(
			mat0.m[0][0] - mat1.m[0][0], mat0.m[0][1] - mat1.m[0][1], mat0.m[0][2] - mat1.m[0][2], mat0.m[0][3] - mat1.m[0][3],
			mat0.m[1][0] - mat1.m[1][0], mat0.m[1][1] - mat1.m[1][1], mat0.m[1][2] - mat1.m[1][2], mat0.m[1][3] - mat1.m[1][3],
			mat0.m[2][0] - mat1.m[2][0], mat0.m[2][1] - mat1.m[2][1], mat0.m[2][2] - mat1.m[2][2], mat0.m[2][3] - mat1.m[2][3],
			mat0.m[3][0] - mat1.m[3][0], mat0.m[3][1] - mat1.m[3][1], mat0.m[3][2] - mat1.m[3][2], mat0.m[3][3] - mat1.m[3][3]);
	}
#pragma endregion

#pragma region Mat4x4 different type operation *
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator*(const T&n, const Mat4x4<U>& mat)
	{
		return Mat4x4<Float>(
			n * mat.m[0][0], n * mat.m[0][1], n * mat.m[0][2], n * mat.m[0][3],
			n * mat.m[1][0], n * mat.m[1][1], n * mat.m[1][2], n * mat.m[1][3],
			n * mat.m[2][0], n * mat.m[2][1], n * mat.m[2][2], n * mat.m[2][3],
			n * mat.m[3][0], n * mat.m[3][1], n * mat.m[3][2], n * mat.m[3][3]);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator*(const Mat4x4<U>& mat, const T&n)
	{
		return Mat4x4<Float>(
			mat.m[0][0] * n, mat.m[0][1] * n, mat.m[0][2] * n, mat.m[0][3] * n,
			mat.m[1][0] * n, mat.m[1][1] * n, mat.m[1][2] * n, mat.m[1][3] * n,
			mat.m[2][0] * n, mat.m[2][1] * n, mat.m[2][2] * n, mat.m[2][3] * n,
			mat.m[3][0] * n, mat.m[3][1] * n, mat.m[3][2] * n, mat.m[3][3] * n);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator*(const Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		Mat4x4<Float> result;
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				result.m[i][j] = dot(mat0.GetRow(i), mat1.GetColumn(j));
			}
		}
		return result;
	}
#pragma endregion

#pragma region Mat4x4 different type operation /
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator/(const T&n, const Mat4x4<U>& mat)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				CHECK(mat.m[i][j] != 0, "Different type Mat4x4 operation /(n, Mat4x4 mat) error: one component of the mat can not be 0!");
			}
		}
		return Mat4x4<Float>(
			n / mat.m[0][0], n / mat.m[0][1], n / mat.m[0][2], n / mat.m[0][3],
			n / mat.m[1][0], n / mat.m[1][1], n / mat.m[1][2], n / mat.m[1][3],
			n / mat.m[2][0], n / mat.m[2][1], n / mat.m[2][2], n / mat.m[2][3],
			n / mat.m[3][0], n / mat.m[3][1], n / mat.m[3][2], n / mat.m[3][3]);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator/(const Mat4x4<U>& mat, const T&n)
	{
		CHECK(n != 0, "Different type Mat4x4 operation /=(Mat4x4 mat, n) error: n can not be 0!");
		return Mat4x4<Float>(
			mat.m[0][0] / n, mat.m[0][1] / n, mat.m[0][2] / n, mat.m[0][3] / n,
			mat.m[1][0] / n, mat.m[1][1] / n, mat.m[1][2] / n, mat.m[1][3] / n,
			mat.m[2][0] / n, mat.m[2][1] / n, mat.m[2][2] / n, mat.m[2][3] / n,
			mat.m[3][0] / n, mat.m[3][1] / n, mat.m[3][2] / n, mat.m[3][3] / n);
	}
	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator/(const Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				CHECK(mat1.m[i][j] != 0, "Different type Mat4x4 operation /(Mat4x4 mat0, Mat4x4 mat1) error: one component of the mat can not be 0!");
			}
		}
		return Mat4x4<Float>(
			mat0.m[0][0] / mat1.m[0][0], mat0.m[0][1] / mat1.m[0][1], mat0.m[0][2] / mat1.m[0][2], mat0.m[0][3] / mat1.m[0][3],
			mat0.m[1][0] / mat1.m[1][0], mat0.m[1][1] / mat1.m[1][1], mat0.m[1][2] / mat1.m[1][2], mat0.m[1][3] / mat1.m[1][3],
			mat0.m[2][0] / mat1.m[2][0], mat0.m[2][1] / mat1.m[2][1], mat0.m[2][2] / mat1.m[2][2], mat0.m[2][3] / mat1.m[2][3],
			mat0.m[3][0] / mat1.m[3][0], mat0.m[3][1] / mat1.m[3][1], mat0.m[3][2] / mat1.m[3][2], mat0.m[3][3] / mat1.m[3][3]);
	}
#pragma endregion

#pragma endregion

#pragma region mat4x4 marco
#define Mat4x4_identity Mat4x4<Int>(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
	//#define Mat4x4_identity Mat4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
#pragma endregion

	template<typename T>
	__duel__ void logData(const Mat4x4<T>& mat)
	{
		const custd::OStream os;
		for (Int i = 0; i < 4; i++)
		{
			for (Int j = 0; j < 4; j++)
			{
				os << mat.m[i][j] << "\t";
			}
			os << custd::endl;
		}
	}

#pragma endregion

#pragma region different class math operation

#pragma region Point3-Vec3
	template<typename T>
	__duel__ const Point3<T> operator+(const Point3<T>& p, const Vec3<T>& v)
	{
		return Point3<T>(p.x + v.x, p.y + v.y, p.z + v.z);
	}

	template<typename T, typename U>
	__duel__ const Point3<Float> operator+(const Point3<T>& p, const Vec3<U>& v)
	{
		return Point3<Float>(p.x + v.x, p.y + v.y, p.z + v.z);
	}

#pragma endregion

#pragma region Mat3x3-Vec3
	template<typename T>
	__duel__ const Vec3<T> operator*(const Mat3x3<T>& mat, const Vec3<T>& v)
	{
		Vec3<T> result;
		for (Int i = 0; i < 3; i++)
		{
			result[i] = dot(mat.GetRow(i), v);
		}
		return result;
	}
#pragma endregion

#pragma region Mat4x4-Vec4
	template<typename T>
	__duel__ const Vec4<T> operator*(const Mat4x4<T>& mat, const Vec4<T>& v)
	{
		Vec4<T> result;
		for (Int i = 0; i < 4; i++)
		{
			result[i] = dot(mat.GetRow(i), v);
		}
		return result;
	}
#pragma endregion

#pragma region Quaternion-Vec4
	template<typename T>
	__duel__ const Vec3<T> applyQuaTransform(const Quaternion<T>& qua, const Vec3<T>& v)
	{
		Vec3<T> result;
		Quaternion<T> pQua(v.x, v.y, v.z, 0);
		Quaternion<T> quaConj = conjugate(qua);
		Quaternion<T> pRes = qua * pQua * quaConj;
		result.x = pRes.x;
		result.y = pRes.y;
		result.z = pRes.z;
		return result;
	}

	template<typename T>
	__duel__ const Vec3f applyInvQuaTransform(const Quaternion<T>& qua, const Vec3<T>& v)
	{
		Vec3<T> result;
		Quaternion<T> pQua(v.x, v.y, v.z, 0);
		Quaternion<T> quaConj = conjugate(qua);
		Quaternion<T> pRes = quaConj* pQua * pQua;
		result.x = pRes.x;
		result.y = pRes.y;
		result.z = pRes.z;
		return result;
	}

#pragma endregion


#pragma region Quaternion-Vec4
	template<typename T>
	__duel__ const Vec4<T> applyQuaTransform(const Quaternion<T>& qua, const Vec4<T>& v)
	{
		Vec4<T> result;
		Quaternion<T> pQua(v.x, v.y, v.z, v.w);
		Quaternion<T> quaConj = conjugate(qua);
		Quaternion<T> pRes = qua * pQua * quaConj;
		result.x = pRes.x;
		result.y = pRes.y;
		result.z = pRes.z;
		result.w = pRes.w;
		return result;
	}

#pragma endregion

#pragma region

#pragma region  other utilities
#pragma endregion

	class Transform
	{
	public:
		Vec3f scale;
		Quaternionf rotation;
		Vec3f translation;
	public:

	};
}



#endif // !__CUDA3DMATH__CUH__

