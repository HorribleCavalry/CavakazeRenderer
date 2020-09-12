#ifndef __CUDA3DMATH__CUH__
#define __CUDA3DMATH__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "../CudaSTD/cuvector.cuh"
#include "../CudaSTD/cuiostream.cuh"
#include <cuda/std/type_traits>
namespace CUM
{
#define LogData(data) logData(data)

#pragma region vec2
	template<typename T>
	class vec2
	{
	public:
		T x, y;
	public:
		__duel__ vec2() :x(0), y(0) {}
		__duel__ vec2(const T& n) :x(n), y(n) {}
		__duel__ vec2(const T& _x, const T& _y) :x(_x), y(_y) {}
		__duel__ vec2(const vec2<T>& v) : x(v.x), y(v.y) {}
		template<typename U>
		__duel__ explicit vec2(const vec2<U>& v) : x(v.x), y(v.y) {}
		__duel__ ~vec2() {}
	public:
		__duel__ const vec2& operator=(const vec2<int>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 1, "The <idx> in vec2<T>::operator[idx] is illegal!");
			return idx == 0 ? x : y;
		}
	};

	typedef vec2<Int> vec2i;
	typedef vec2<Float> vec2f;

#pragma region vec2 vector operation

	template<typename T>
	__duel__ const T dot(const vec2<T>& v0, const vec2<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y;
	}

	template<typename T, typename U>
	__duel__ Float dot(const vec2<T>& v0, const vec2<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y;
	}

#pragma endregion

#pragma region vec2 same type operation

#pragma region vec2 same type operation +

	template<typename T>
	__duel__ const vec2<T> operator+(const T& n, const vec2<T>& v)
	{
		return vec2<T>(n + v.x, n + v.y);
	}
	template<typename T>
	__duel__ const vec2<T> operator+(const vec2<T>& v, const T& n)
	{
		return vec2<T>(v.x + n, v.y + n);
	}
	template<typename T>
	__duel__ const vec2<T> operator+(const vec2<T>& v0, const vec2<T>& v1)
	{
		return vec2<T>(v0.x + v1.x, v0.y + v1.y);
	}

	template<typename T, typename U>
	__duel__ const vec2<T>& operator+=(vec2<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec2<T>& operator+=(vec2<T>& v0, const vec2<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		return v0;
	}

	__duel__ const vec2<Int>& operator+=(vec2<Int>& v, const Float& n) = delete;
	__duel__ const vec2<Int>& operator+=(vec2<Int>& v0, const vec2<Float>& v1) = delete;


#pragma endregion

#pragma region vec2 same type operation -

	template<typename T>
	__duel__ const vec2<T> operator-(const T& n, const vec2<T>& v)
	{
		return vec2<T>(n - v.x, n - v.y);
	}
	template<typename T>
	__duel__ const vec2<T> operator-(const vec2<T>& v, const T& n)
	{
		return vec2<T>(v.x - n, v.y - n);
	}
	template<typename T>
	__duel__ const vec2<T> operator-(const vec2<T>& v0, const vec2<T>& v1)
	{
		return vec2<T>(v0.x - v1.x, v0.y - v1.y);
	}

	template<typename T>
	__duel__ const vec2<T>& operator-=(vec2<T>& v, const T& n)
	{
		v.x -= n;
		v.y -= n;
		return v;
	}
	template<typename T>
	__duel__ const vec2<T>& operator-=(vec2<T>& v0, const vec2<T>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		return v0;
	}

	__duel__ const vec2<Int>& operator-=(vec2<Int>& v, const Float& n) = delete;
	__duel__ const vec2<Int>& operator-=(vec2<Int>& v0, const vec2<Float>& v1) = delete;

#pragma endregion

#pragma region vec2 same type operation *

	template<typename T>
	__duel__ const vec2<T> operator*(const T& n, const vec2<T>& v)
	{
		return vec2<T>(n * v.x, n * v.y);
	}
	template<typename T>
	__duel__ const vec2<T> operator*(const vec2<T>& v, const T& n)
	{
		return vec2<T>(v.x * n, v.y * n);
	}
	template<typename T>
	__duel__ const vec2<T> operator*(const vec2<T>& v0, const vec2<T>& v1)
	{
		return vec2<T>(v0.x * v1.x, v0.y * v1.y);
	}

	template<typename T>
	__duel__ const vec2<T>& operator*=(vec2<T>& v, const T& n)
	{
		v.x *= n;
		v.y *= n;
		return v;
	}
	template<typename T>
	__duel__ const vec2<T>& operator*=(vec2<T>& v0, const vec2<T>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		return v0;
	}

	__duel__ const vec2<Int>& operator*=(vec2<Int>& v, const Float& n) = delete;
	__duel__ const vec2<Int>& operator*=(vec2<Int>& v0, const vec2<Float>& v1) = delete;

#pragma endregion

#pragma region vec2 same type operation /

	template<typename T>
	__duel__ const vec2<T> operator/(const T& n, const vec2<T>& v)
	{
		CHECK(v.x != 0, "Same type vec2 operator/ error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type vec2 operator/ error: v.y can not be 0!");
		return vec2<T>(n / v.x, n / v.y);
	}
	template<typename T>
	__duel__ const vec2<T> operator/(const vec2<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type vec2 operator/ error: n can not be 0!");
		return vec2<T>(v.x / n, v.y / n);
	}
	template<typename T>
	__duel__ const vec2<T> operator/(const vec2<T>& v0, const vec2<T>& v1)
	{
		CHECK(v1.x != 0, "Same type vec2 operator/ error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type vec2 operator/ error: v1.y can not be 0!");
		return vec2<T>(v0.x / v1.x, v0.y / v1.y);
	}

	template<typename T>
	__duel__ const vec2<T>& operator/=(vec2<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type vec2 operator/= error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		return v;
	}
	template<typename T>
	__duel__ const vec2<T>& operator/=(vec2<T>& v0, const vec2<T>& v1)
	{
		CHECK(v1.x != 0, "Same type vec2 operator/= error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type vec2 operator/= error: v1.y can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		return v0;
	}

	__duel__ const vec2<Int>& operator/=(vec2<Int>& v, const Float& n) = delete;
	__duel__ const vec2<Int>& operator/=(vec2<Int>& v0, const vec2<Float>& v1) = delete;

#pragma endregion

#pragma endregion

#pragma region vec2 different type operation

#pragma region vec2 different type operation +

	template<typename T, typename U>
	__duel__ const vec2<Float> operator+(const T& n, const vec2<U>& v)
	{
		return vec2<Float>(n + v.x, n + v.y);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator+(const vec2<T>& v, const U& n)
	{
		return vec2<Float>(v.x + n, v.y + n);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator+(const vec2<T>& v0, const vec2<U>& v1)
	{
		return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
	}

#pragma endregion

#pragma region vec2 different type operation -

	template<typename T, typename U>
	__duel__ const vec2<Float> operator-(const T& n, const vec2<U>& v)
	{
		return vec2<Float>(n - v.x, n - v.y);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator-(const vec2<T>& v, const U& n)
	{
		return vec2<Float>(v.x - n, v.y - n);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator-(const vec2<T>& v0, const vec2<U>& v1)
	{
		return vec2<Float>(v0.x - v1.x, v0.y - v1.y);
	}

#pragma endregion

#pragma region vec2 different type operation *

	template<typename T, typename U>
	__duel__ const vec2<Float> operator*(const T& n, const vec2<U>& v)
	{
		return vec2<Float>(n * v.x, n * v.y);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator*(const vec2<T>& v, const U& n)
	{
		return vec2<Float>(v.x * n, v.y * n);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator*(const vec2<T>& v0, const vec2<U>& v1)
	{
		return vec2<Float>(v0.x * v1.x, v0.y * v1.y);
	}

#pragma endregion

#pragma region vec2 different type operation /

	template<typename T, typename U>
	__duel__ const vec2<Float> operator/(const T& n, const vec2<U>& v)
	{
		CHECK(v.x != 0, "vec2<Float> operation /(n,vec2 v): v.x can not be zero.");
		CHECK(v.y != 0, "vec2<Float> operation /(n,vec2 v): v.y can not be zero.");
		return vec2<Float>(n / v.x, n / v.y);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator/(const vec2<T>& v, const U& n)
	{
		CHECK(v.y != 0, "vec2<Float> operation /(vec2 v,n): n can not be zero.");
		return vec2<Float>(v.x / n, v.y / n);
	}
	template<typename T, typename U>
	__duel__ const vec2<Float> operator/(const vec2<T>& v0, const vec2<U>& v1)
	{
		CHECK(v1.x != 0, "vec2<Float> operation /(vec2 v0,vec2 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "vec2<Float> operation /(vec2 v0,vec2 v1): v1.y can not be zero.");
		return vec2<Float>(v0.x / v1.x, v0.y / v1.y);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const vec2<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << custd::endl;
	}

#pragma endregion

#pragma region vec3
	template<typename T>
	class vec3
	{
	public:
		T x, y, z;
	public:
		__duel__ vec3() :x(0), y(0), z(0) {}
		__duel__ vec3(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}
		__duel__ vec3(const T& n) : x(n), y(n), z(n) {}
		__duel__ vec3(const vec3<T>& v) : x(v.x), y(v.y), z(v.z) {}
		template<typename U>
		__duel__ explicit vec3(const vec3<U>& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ ~vec3() {}
	public:
		__duel__ const vec3& operator=(const vec3<int>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "The <idx> in vec3<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			default: CHECK(false, "Can not run vec3::operator[idx]: switch::default."); break;
			}
		}
	};

	typedef vec3<Int> vec3i;
	typedef vec3<Float> vec3f;

#pragma region vec3 vector operation

	template<typename T>
	__duel__ const T dot(const vec3<T>& v0, const vec3<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z;
	}

	template<typename T, typename U>
	__duel__ Float dot(const vec3<T>& v0, const vec3<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z;
	}

	template<typename T>
	__duel__ const vec3<T> cross(const vec3<T>& v0, const vec3<T>& v1)
	{
		return vec3<T>(v0.y*v1.z - v0.z*v1.y, v0.z*v1.x - v0.x*v1.z, v0.x*v1.y - v0.y*v1.x);
	}

	template<typename T, typename U>
	__duel__ const vec3<Float> cross(const vec3<T>& v0, const vec3<U>& v1)
	{
		return vec3<Float>(v0.y*v1.z - v0.z*v1.y, v0.z*v1.x - v0.x*v1.z, v0.x*v1.y - v0.y*v1.x);
	}

#pragma endregion

#pragma region vec3 same type operation

#pragma region vec3 same type operation +

	template<typename T>
	__duel__ const vec3<T> operator+(const T& n, const vec3<T>& v)
	{
		return vec3<T>(n + v.x, n + v.y, n + v.z);
	}
	template<typename T>
	__duel__ const vec3<T> operator+(const vec3<T>& v, const T& n)
	{
		return vec3<T>(v.x + n, v.y + n, v.z + n);
	}
	template<typename T>
	__duel__ const vec3<T> operator+(const vec3<T>& v0, const vec3<T>& v1)
	{
		return vec3<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	template<typename T, typename U>
	__duel__ const vec3<T>& operator+=(vec3<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec3<T>& operator+=(vec3<T>& v0, const vec3<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		return v0;
	}

	__duel__ const vec3<Int>& operator+=(vec3<Int>& v, const Float& n) = delete;
	__duel__ const vec3<Int>& operator+=(vec3<Int>& v0, const vec3<Float>& v1) = delete;


#pragma endregion

#pragma region vec3 same type operation -

	template<typename T>
	__duel__ const vec3<T> operator-(const T& n, const vec3<T>& v)
	{
		return vec3<T>(n - v.x, n - v.y, n - v.z);
	}
	template<typename T>
	__duel__ const vec3<T> operator-(const vec3<T>& v, const T& n)
	{
		return vec3<T>(v.x - n, v.y - n, v.z - n);
	}
	template<typename T>
	__duel__ const vec3<T> operator-(const vec3<T>& v0, const vec3<T>& v1)
	{
		return vec3<T>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

	template<typename T, typename U>
	__duel__ const vec3<T>& operator-=(vec3<T>& v, const U& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec3<T>& operator-=(vec3<T>& v0, const vec3<U>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		return v0;
	}

	__duel__ const vec3<Int>& operator-=(vec3<Int>& v, const Float& n) = delete;
	__duel__ const vec3<Int>& operator-=(vec3<Int>& v0, const vec3<Float>& v1) = delete;


#pragma endregion

#pragma region vec3 same type operation *

	template<typename T>
	__duel__ const vec3<T> operator*(const T& n, const vec3<T>& v)
	{
		return vec3<T>(n * v.x, n * v.y, n * v.z);
	}
	template<typename T>
	__duel__ const vec3<T> operator*(const vec3<T>& v, const T& n)
	{
		return vec3<T>(v.x * n, v.y * n, v.z * n);
	}
	template<typename T>
	__duel__ const vec3<T> operator*(const vec3<T>& v0, const vec3<T>& v1)
	{
		return vec3<T>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	template<typename T, typename U>
	__duel__ const vec3<T>& operator*=(vec3<T>& v, const U& n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec3<T>& operator*=(vec3<T>& v0, const vec3<U>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		return v0;
	}

	__duel__ const vec3<Int>& operator*=(vec3<Int>& v, const Float& n) = delete;
	__duel__ const vec3<Int>& operator*=(vec3<Int>& v0, const vec3<Float>& v1) = delete;


#pragma endregion

#pragma region vec3 same type operation /

	template<typename T>
	__duel__ const vec3<T> operator/(const T& n, const vec3<T>& v)
	{
		CHECK(v.x != 0, "Same type vec3 operator/(n,vec3 v) error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type vec3 operator/(n,vec3 v) error: v.y can not be 0!");
		CHECK(v.z != 0, "Same type vec3 operator/(n,vec3 v) error: v.z can not be 0!");
		return vec3<T>(n / v.x, n / v.y, n / v.z);
	}
	template<typename T>
	__duel__ const vec3<T> operator/(const vec3<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type vec3 operator/(vec3 v, n) error: n can not be 0!");
		return vec3<T>(v.x / n, v.y / n, v.z / n);
	}
	template<typename T>
	__duel__ const vec3<T> operator/(const vec3<T>& v0, const vec3<T>& v1)
	{
		CHECK(v1.x != 0, "Same type vec3 operator/(n,vec3 v) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type vec3 operator/(n,vec3 v) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type vec3 operator/(n,vec3 v) error: v1.z can not be 0!");
		return vec3<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	template<typename T, typename U>
	__duel__ const vec3<T>& operator/=(vec3<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type vec3 operator/=(vec3 v, n) error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		v.z /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec3<T>& operator/=(vec3<T>& v0, const vec3<U>& v1)
	{
		CHECK(v1.x != 0, "Same type vec3 operator/=(vec3 v0,vec3 v1) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type vec3 operator/=(vec3 v0,vec3 v1) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type vec3 operator/=(vec3 v0,vec3 v1) error: v1.z can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}

	__duel__ const vec3<Int>& operator/=(vec3<Int>& v, const Float& n) = delete;
	__duel__ const vec3<Int>& operator/=(vec3<Int>& v0, const vec3<Float>& v1) = delete;


#pragma endregion


#pragma endregion

#pragma region vec3 different type operation

#pragma region vec3 different type operation +

	template<typename T, typename U>
	__duel__ const vec3<Float> operator+(const T& n, const vec3<U>& v)
	{
		return vec3<Float>(n + v.x, n + v.y);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator+(const vec3<T>& v, const U& n)
	{
		return vec3<Float>(v.x + n, v.y + n);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator+(const vec3<T>& v0, const vec3<U>& v1)
	{
		return vec3<Float>(v0.x + v1.x, v0.y + v1.y);
	}

#pragma endregion

#pragma region vec3 different type operation -

	template<typename T, typename U>
	__duel__ const vec3<Float> operator-(const T& n, const vec3<U>& v)
	{
		return vec3<Float>(n - v.x, n - v.y);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator-(const vec3<T>& v, const U& n)
	{
		return vec3<Float>(v.x - n, v.y - n);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator-(const vec3<T>& v0, const vec3<U>& v1)
	{
		return vec3<Float>(v0.x - v1.x, v0.y - v1.y);
	}

#pragma endregion

#pragma region vec3 different type operation *

	template<typename T, typename U>
	__duel__ const vec3<Float> operator*(const T& n, const vec3<U>& v)
	{
		return vec3<Float>(n * v.x, n * v.y);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator*(const vec3<T>& v, const U& n)
	{
		return vec3<Float>(v.x * n, v.y * n);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator*(const vec3<T>& v0, const vec3<U>& v1)
	{
		return vec3<Float>(v0.x * v1.x, v0.y * v1.y);
	}

#pragma endregion

#pragma region vec3 different type operation /

	template<typename T, typename U>
	__duel__ const vec3<Float> operator/(const T& n, const vec3<U>& v)
	{
		CHECK(v.x != 0, "vec3<Float> operation /(n, vec3 v1): v1.x can not be zero.");
		CHECK(v.y != 0, "vec3<Float> operation /(n, vec3 v1): v1.y can not be zero.");
		CHECK(v.z != 0, "vec3<Float> operation /(n, vec3 v1): v1.z can not be zero.");
		return vec3<Float>(n / v.x, n / v.y);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator/(const vec3<T>& v, const U& n)
	{
		CHECK(n != 0, "vec3<Float> operation /(vec3 v, n): n can not be zero.");
		return vec3<Float>(v.x / n, v.y / n);
	}
	template<typename T, typename U>
	__duel__ const vec3<Float> operator/(const vec3<T>& v0, const vec3<U>& v1)
	{
		CHECK(v1.x != 0, "vec3<Float> operation /(vec3 v0, vec3 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "vec3<Float> operation /(vec3 v0, vec3 v1): v1.y can not be zero.");
		CHECK(v1.z != 0, "vec3<Float> operation /(vec3 v0, vec3 v1): v1.z can not be zero.");
		return vec3<Float>(v0.x / v1.x, v0.y / v1.y);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const vec3<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << "\t" << v.z << custd::endl;
	}

#pragma endregion

#pragma region vec4
	template<typename T>
	class vec4
	{
	public:
		T x, y, z, w;
	public:
		__duel__ vec4() :x(0), y(0), z(0), w(0) {}
		__duel__ vec4(const T& _x, const T& _y, const T& _z, const T& _w) : x(_x), y(_y), z(_z), w(_w) {}
		__duel__ vec4(const T& n) : x(n), y(n), z(n), w(n) {}
		__duel__ vec4(const vec4<T>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		template<typename U>
		__duel__ explicit vec4(const vec4<U>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		__duel__ ~vec4() {}
	public:
		__duel__ const vec4& operator=(const vec4<int>& v)
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
			CHECK(idx >= 0 && idx <= 3, "The <idx> in vec4<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			case 3: return w; break;
			default: CHECK(false, "Can not run vec4::operator[idx]: switch::default."); break;
			}
		}
	};

	typedef vec4<Int> vec4i;
	typedef vec4<Float> vec4f;

#pragma region vec4 vector operation

	template<typename T>
	__duel__ const T dot(const vec4<T>& v0, const vec4<T>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z + v0.w * v1.w;
	}

	template<typename T, typename U>
	__duel__ Float dot(const vec4<T>& v0, const vec4<U>& v1)
	{
		return v0.x*v1.x + v0.y * v1.y + v0.z*v1.z + v0.w*v1.w;
	}

#pragma endregion

#pragma region vec4 same type operation

#pragma region vec4 same type operation +

	template<typename T>
	__duel__ const vec4<T> operator+(const T& n, const vec4<T>& v)
	{
		return vec4<T>(n + v.x, n + v.y, n + v.z, n + v.w);
	}
	template<typename T>
	__duel__ const vec4<T> operator+(const vec4<T>& v, const T& n)
	{
		return vec4<T>(v.x + n, v.y + n, v.z + n, v.w + n);
	}
	template<typename T>
	__duel__ const vec4<T> operator+(const vec4<T>& v0, const vec4<T>& v1)
	{
		return vec4<T>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
	}

	template<typename T, typename U>
	__duel__ const vec4<T>& operator+=(vec4<T>& v, const U& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		v.w += n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec4<T>& operator+=(vec4<T>& v0, const vec4<U>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		v0.w += v1.w;
		return v0;
	}

	__duel__ const vec4<Int>& operator+=(vec4<Int>& v, const Float& n) = delete;
	__duel__ const vec4<Int>& operator+=(vec4<Int>& v0, const vec4<Float>& v1) = delete;


#pragma endregion

#pragma region vec4 same type operation -

	template<typename T>
	__duel__ const vec4<T> operator-(const T& n, const vec4<T>& v)
	{
		return vec4<T>(n - v.x, n - v.y, n - v.z, n - v.w);
	}
	template<typename T>
	__duel__ const vec4<T> operator-(const vec4<T>& v, const T& n)
	{
		return vec4<T>(v.x - n, v.y - n, v.z - n, v.w - n);
	}
	template<typename T>
	__duel__ const vec4<T> operator-(const vec4<T>& v0, const vec4<T>& v1)
	{
		return vec4<T>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
	}

	template<typename T, typename U>
	__duel__ const vec4<T>& operator-=(vec4<T>& v, const U& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		v.w -= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec4<T>& operator-=(vec4<T>& v0, const vec4<U>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		v0.w -= v1.w;
		return v0;
	}

	__duel__ const vec4<Int>& operator-=(vec4<Int>& v, const Float& n) = delete;
	__duel__ const vec4<Int>& operator-=(vec4<Int>& v0, const vec4<Float>& v1) = delete;


#pragma endregion

#pragma region vec4 same type operation *

	template<typename T>
	__duel__ const vec4<T> operator*(const T& n, const vec4<T>& v)
	{
		return vec4<T>(n * v.x, n * v.y, n * v.z, n * v.w);
	}
	template<typename T>
	__duel__ const vec4<T> operator*(const vec4<T>& v, const T& n)
	{
		return vec4<T>(v.x * n, v.y * n, v.z * n,v.w * n);
	}
	template<typename T>
	__duel__ const vec4<T> operator*(const vec4<T>& v0, const vec4<T>& v1)
	{
		return vec4<T>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
	}

	template<typename T, typename U>
	__duel__ const vec4<T>& operator*=(vec4<T>& v, const U& n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		v.w *= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec4<T>& operator*=(vec4<T>& v0, const vec4<U>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		v0.w *= v1.w;
		return v0;
	}

	__duel__ const vec4<Int>& operator*=(vec4<Int>& v, const Float& n) = delete;
	__duel__ const vec4<Int>& operator*=(vec4<Int>& v0, const vec4<Float>& v1) = delete;


#pragma endregion

#pragma region vec4 same type operation /

	template<typename T>
	__duel__ const vec4<T> operator/(const T& n, const vec4<T>& v)
	{
		CHECK(v.x != 0, "Same type vec4 operator/(n,vec4 v) error: v.x can not be 0!");
		CHECK(v.y != 0, "Same type vec4 operator/(n,vec4 v) error: v.y can not be 0!");
		CHECK(v.z != 0, "Same type vec4 operator/(n,vec4 v) error: v.z can not be 0!");
		CHECK(v.w != 0, "Same type vec4 operator/(n,vec4 v) error: v.w can not be 0!");
		return vec4<T>(n / v.x, n / v.y, n / v.z);
	}
	template<typename T>
	__duel__ const vec4<T> operator/(const vec4<T>& v, const T& n)
	{
		CHECK(n != 0, "Same type vec4 operator/(vec4 v, n) error: n can not be 0!");
		return vec4<T>(v.x / n, v.y / n, v.z / n);
	}
	template<typename T>
	__duel__ const vec4<T> operator/(const vec4<T>& v0, const vec4<T>& v1)
	{
		CHECK(v1.x != 0, "Same type vec4 operator/(n,vec4 v) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type vec4 operator/(n,vec4 v) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type vec4 operator/(n,vec4 v) error: v1.z can not be 0!");
		CHECK(v1.w != 0, "Same type vec4 operator/(n,vec4 v) error: v1.w can not be 0!");
		return vec4<T>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	template<typename T, typename U>
	__duel__ const vec4<T>& operator/=(vec4<T>& v, const U& n)
	{
		CHECK(n != 0, "Same type vec4 operator/=(vec4 v, n) error: n can not be 0!");
		v.x /= n;
		v.y /= n;
		v.z /= n;
		return v;
	}
	template<typename T, typename U>
	__duel__ const vec4<T>& operator/=(vec4<T>& v0, const vec4<U>& v1)
	{
		CHECK(v1.x != 0, "Same type vec4 operator/=(vec4 v0,vec4 v1) error: v1.x can not be 0!");
		CHECK(v1.y != 0, "Same type vec4 operator/=(vec4 v0,vec4 v1) error: v1.y can not be 0!");
		CHECK(v1.z != 0, "Same type vec4 operator/=(vec4 v0,vec4 v1) error: v1.z can not be 0!");
		CHECK(v1.w != 0, "Same type vec4 operator/=(vec4 v0,vec4 v1) error: v1.w can not be 0!");
		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}

	__duel__ const vec4<Int>& operator/=(vec4<Int>& v, const Float& n) = delete;
	__duel__ const vec4<Int>& operator/=(vec4<Int>& v0, const vec4<Float>& v1) = delete;


#pragma endregion


#pragma endregion

#pragma region vec4 different type operation

#pragma region vec4 different type operation +

	template<typename T, typename U>
	__duel__ const vec4<Float> operator+(const T& n, const vec4<U>& v)
	{
		return vec4<Float>(n + v.x, n + v.y);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator+(const vec4<T>& v, const U& n)
	{
		return vec4<Float>(v.x + n, v.y + n);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator+(const vec4<T>& v0, const vec4<U>& v1)
	{
		return vec4<Float>(v0.x + v1.x, v0.y + v1.y);
	}

#pragma endregion

#pragma region vec4 different type operation -

	template<typename T, typename U>
	__duel__ const vec4<Float> operator-(const T& n, const vec4<U>& v)
	{
		return vec4<Float>(n - v.x, n - v.y);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator-(const vec4<T>& v, const U& n)
	{
		return vec4<Float>(v.x - n, v.y - n);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator-(const vec4<T>& v0, const vec4<U>& v1)
	{
		return vec4<Float>(v0.x - v1.x, v0.y - v1.y);
	}

#pragma endregion

#pragma region vec4 different type operation *

	template<typename T, typename U>
	__duel__ const vec4<Float> operator*(const T& n, const vec4<U>& v)
	{
		return vec4<Float>(n * v.x, n * v.y);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator*(const vec4<T>& v, const U& n)
	{
		return vec4<Float>(v.x * n, v.y * n);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator*(const vec4<T>& v0, const vec4<U>& v1)
	{
		return vec4<Float>(v0.x * v1.x, v0.y * v1.y);
	}

#pragma endregion

#pragma region vec4 different type operation /

	template<typename T, typename U>
	__duel__ const vec4<Float> operator/(const T& n, const vec4<U>& v)
	{
		CHECK(v.x != 0, "vec4<Float> operation /(n, vec4 v1): v1.x can not be zero.");
		CHECK(v.y != 0, "vec4<Float> operation /(n, vec4 v1): v1.y can not be zero.");
		CHECK(v.z != 0, "vec4<Float> operation /(n, vec4 v1): v1.z can not be zero.");
		CHECK(v.w != 0, "vec4<Float> operation /(n, vec4 v1): v1.w can not be zero.");
		return vec4<Float>(n / v.x, n / v.y);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator/(const vec4<T>& v, const U& n)
	{
		CHECK(n != 0, "vec4<Float> operation /(vec4 v, n): n can not be zero.");
		return vec4<Float>(v.x / n, v.y / n);
	}
	template<typename T, typename U>
	__duel__ const vec4<Float> operator/(const vec4<T>& v0, const vec4<U>& v1)
	{
		CHECK(v1.x != 0, "vec4<Float> operation /(vec4 v0, vec4 v1): v1.x can not be zero.");
		CHECK(v1.y != 0, "vec4<Float> operation /(vec4 v0, vec4 v1): v1.y can not be zero.");
		CHECK(v1.z != 0, "vec4<Float> operation /(vec4 v0, vec4 v1): v1.z can not be zero.");
		CHECK(v1.w != 0, "vec4<Float> operation /(vec4 v0, vec4 v1): v1.w can not be zero.");
		return vec4<Float>(v0.x / v1.x, v0.y / v1.y);
	}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const vec4<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << "\t" << v.z << "\t" << v.w << custd::endl;
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
		const vec3<T> GetRow(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "Mat3x3::GetRow(idx) error: idx is out of range!");
			return vec3<T>(m[idx][0], m[idx][1], m[idx][2]);
		}
		const vec3<T> GetColumn(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "Mat3x3::GetColumn(idx) error: idx is out of range!");
			return vec3<T>(m[0][idx], m[1][idx], m[2][idx]);
		}
		const vec3<T> GetDiag()
		{
			return vec3<T>(m[0][0], m[1][1], m[2][2]);
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
		return Mat3x3<T>(
			mat0.m[0][0] * mat1.m[0][0], mat0.m[0][1] * mat1.m[0][1], mat0.m[0][2] * mat1.m[0][2],
			mat0.m[1][0] * mat1.m[1][0], mat0.m[1][1] * mat1.m[1][1], mat0.m[1][2] * mat1.m[1][2],
			mat0.m[2][0] * mat1.m[2][0], mat0.m[2][1] * mat1.m[2][1], mat0.m[2][2] * mat1.m[2][2]);
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
		return Mat3x3<Float>(
			mat0.m[0][0] * mat1.m[0][0], mat0.m[0][1] * mat1.m[0][1], mat0.m[0][2] * mat1.m[0][2],
			mat0.m[1][0] * mat1.m[1][0], mat0.m[1][1] * mat1.m[1][1], mat0.m[1][2] * mat1.m[1][2],
			mat0.m[2][0] * mat1.m[2][0], mat0.m[2][1] * mat1.m[2][1], mat0.m[2][2] * mat1.m[2][2]);
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
		const custd::Ostream os;
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
		const vec4<T> GetRow(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 3, "Mat4x4::GetRow(idx) error: idx is out of range!");
			return vec4<T>(m[idx][0], m[idx][1], m[idx][2], m[idx][3]);
		}
		const vec4<T> GetColumn(const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 3, "Mat4x4::GetColumn(idx) error: idx is out of range!");
			return vec4<T>(m[0][idx], m[1][idx], m[2][idx], m[3][idx]);
		}
		const vec4<T> GetDiag()
		{
			return vec4<T>(m[0][0], m[1][1], m[2][2], m[3][3]);
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
		return Mat4x4<T>(
			mat0.m[0][0] * mat1.m[0][0], mat0.m[0][1] * mat1.m[0][1], mat0.m[0][2] * mat1.m[0][2], mat0.m[0][3] * mat1.m[0][3],
			mat0.m[1][0] * mat1.m[1][0], mat0.m[1][1] * mat1.m[1][1], mat0.m[1][2] * mat1.m[1][2], mat0.m[1][3] * mat1.m[1][3],
			mat0.m[2][0] * mat1.m[2][0], mat0.m[2][1] * mat1.m[2][1], mat0.m[2][2] * mat1.m[2][2], mat0.m[2][3] * mat1.m[2][3],
			mat0.m[3][0] * mat1.m[3][0], mat0.m[3][1] * mat1.m[3][1], mat0.m[3][2] * mat1.m[3][2], mat0.m[3][3] * mat1.m[3][3]);
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
		return Mat4x4<Float>(
			mat0.m[0][0] * mat1.m[0][0], mat0.m[0][1] * mat1.m[0][1], mat0.m[0][2] * mat1.m[0][2], mat0.m[0][3] * mat1.m[0][3],
			mat0.m[1][0] * mat1.m[1][0], mat0.m[1][1] * mat1.m[1][1], mat0.m[1][2] * mat1.m[1][2], mat0.m[1][3] * mat1.m[1][3],
			mat0.m[2][0] * mat1.m[2][0], mat0.m[2][1] * mat1.m[2][1], mat0.m[2][2] * mat1.m[2][2], mat0.m[2][3] * mat1.m[2][3],
			mat0.m[3][0] * mat1.m[3][0], mat0.m[3][1] * mat1.m[3][1], mat0.m[3][2] * mat1.m[3][2], mat0.m[3][3] * mat1.m[3][3]);
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
		const custd::Ostream os;
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
}

#endif // !__CUDA3DMATH__CUH__

