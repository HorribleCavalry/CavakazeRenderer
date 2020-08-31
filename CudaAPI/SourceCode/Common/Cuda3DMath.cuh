#ifndef __CUDA3DMATH__CUH__
#define __CUDA3DMATH__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "../CudaSTD/cuvector.cuh"
namespace CUM
{
#pragma region vec2
	template<typename T>
	class vec2
	{
	public:
		T x, y;
	public:
		__duel__ vec2() :x(0.0), y(0.0) {}
		__duel__ vec2(const vec2<T>& v) : x(v.x), y(v.y) {}
		__duel__ vec2(vec2<T>&& v) : x(v.x), y(v.y) {}
		__duel__ vec2<T>& operator=(const vec2<T>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}
		__duel__ vec2<T>& operator=(vec2<T>&& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}
		__duel__ ~vec2() {}
	public:
		__duel__ vec2(const T& n) : x(n), y(n) {}
		__duel__ vec2(const T& _x, const T& _y) : x(_x), y(_y) {}

		template<typename U>
		__duel__ vec2<T>(const vec2<U>& v) : x(v.x), y(v.y) {}

	public:
		__duel__ T& operator[](const T& idx)
		{
			CHECK(idx >= 0 && idx <= 1, "The <idx> in vec2<T>::operator[idx] is illegal!");
			return idx == 0 ? x : y;
		}
	public:
	};

	template<>
	class vec2<Int>
	{
	public:
		Int x, y;
	public:
		__duel__ vec2() :x(0), y(0) {}
		__duel__ vec2(const vec2<Int>& v) : x(v.x), y(v.y) {}
		__duel__ vec2(vec2<Int>&& v) : x(v.x), y(v.y) {}
		__duel__ const vec2<Int>& operator=(const vec2<Int>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}
		__duel__ const vec2<Int>& operator=(vec2<Int>&& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}
		__duel__ ~vec2() {}
	public:
		__duel__ vec2(const Int& n) : x(n), y(n) {}
		__duel__ vec2(const Int& _x, const Int& _y) : x(_x), y(_y) {}

		__duel__ vec2(const Float& n) = delete;
		__duel__ vec2(const Float& _x, const Float& _y) = delete;

		__duel__ explicit vec2(const vec2<Float>& v) : x(v.x), y(v.y) {}

		__duel__ const vec2<Int>& operator=(const Int& n)
		{
			x = n;
			y = n;
			return *this;
		}
		__duel__ const vec2<Int>& operator=(Int&& n)
		{
			x = n;
			y = n;
			return *this;
		}

		__duel__ const vec2<Int>& operator=(const Float& n) = delete;
		__duel__ const vec2<Int>& operator=(Float&& n) = delete;
	public:
		__duel__ Int& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 1, "The <idx> in vec2<Int>::operator[idx] is illegal!");
			return idx == 0 ? x : y;
		}
	public:

	};

	typedef vec2<Int> vec2i;
	typedef vec2<Float> vec2f;

#pragma region vec2 dot operation

	const Int dot(const vec2<Int>& v0, const vec2<Int>& v1);
	const Float dot(const vec2<Int>& v0, const vec2<Float>& v1);
	const Float dot(const vec2<Float>& v0, const vec2<Int>& v1);
	const Float dot(const vec2<Float>& v0, const vec2<Float>& v1);

	const Int dot(const Int& n, const vec2<Int>& v);
	const Float dot(const Int& n, const vec2<Float>& v);
	const Float dot(const Float& n, const vec2<Int>& v);
	const Float dot(const Float& n, const vec2<Float>& v);

	const Int dot(const vec2<Int>& v, const Int& n);
	const Float dot(const vec2<Int>& v, const Float& n);
	const Float dot(const vec2<Float>& v, const Int& n);
	const Float dot(const vec2<Float>& v, const Float& n);

#pragma endregion

#pragma region vec2i add operation

	__duel__ const vec2<Int> operator+(const Int& n, const vec2<Int>& v);
	__duel__ const vec2<Int> operator+(const vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int> operator+(const vec2<Int>& v0, const vec2<Int>& v1);

	__duel__ const vec2<Int>& operator+=(vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int>& operator+=(vec2<Int>& v0, const vec2<Int>& v1);

#pragma endregion

#pragma region vec2i subtract operation

	__duel__ const vec2<Int> operator-(const Int& n, const vec2<Int>& v);
	__duel__ const vec2<Int> operator-(const vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int> operator-(const vec2<Int>& v0, const vec2<Int>& v1);

	__duel__ const vec2<Int>& operator-=(vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int>& operator-=(vec2<Int>& v0, const vec2<Int>& v1);

#pragma endregion

#pragma region vec2i multiply operation

	__duel__ const vec2<Int> operator*(const Int& n, const vec2<Int>& v);
	__duel__ const vec2<Int> operator*(const vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int> operator*(const vec2<Int>& v0, const vec2<Int>& v1);

	__duel__ const vec2<Int>& operator*=(vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int>& operator*=(vec2<Int>& v0, const vec2<Int>& v1);

#pragma endregion

#pragma region vec2i divide operation

	__duel__ const vec2<Int> operator/(const Int& n, const vec2<Int>& v);
	__duel__ const vec2<Int> operator/(const vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int> operator/(const vec2<Int>& v0, const vec2<Int>& v1);

	__duel__ const vec2<Int>& operator/=(vec2<Int>& v, const Int& n);
	__duel__ const vec2<Int>& operator/=(vec2<Int>& v0, const vec2<Int>& v1);

#pragma endregion

#pragma region vec2f add operation

	__duel__ const vec2<Float> operator+(const Float& n, const vec2<Float>& v);
	__duel__ const vec2<Float> operator+(const vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float> operator+(const vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float> operator+(const vec2<Int>& v0, const vec2<Float>& v1);
	__duel__ const vec2<Float> operator+(const vec2<Float>& v0, const vec2<Float>& v1);

	__duel__ const vec2<Float>& operator+=(vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<Float>& v1);

#pragma endregion

#pragma region vec2f subtract operation

	__duel__ const vec2<Float> operator-(const Float& n, const vec2<Float>& v);
	__duel__ const vec2<Float> operator-(const vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float> operator-(const vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float> operator-(const vec2<Int>& v0, const vec2<Float>& v1);
	__duel__ const vec2<Float> operator-(const vec2<Float>& v0, const vec2<Float>& v1);

	__duel__ const vec2<Float>& operator-=(vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float>& operator-=(vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float>& operator-=(vec2<Float>& v0, const vec2<Float>& v1);

#pragma endregion

#pragma region vec2f multiply operation

	__duel__ const vec2<Float> operator*(const Float& n, const vec2<Float>& v);
	__duel__ const vec2<Float> operator*(const vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float> operator*(const vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float> operator*(const vec2<Int>& v0, const vec2<Float>& v1);
	__duel__ const vec2<Float> operator*(const vec2<Float>& v0, const vec2<Float>& v1);

	__duel__ const vec2<Float>& operator*=(vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float>& operator*=(vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float>& operator*=(vec2<Float>& v0, const vec2<Float>& v1);

#pragma endregion

#pragma region vec2f divide operation

	__duel__ const vec2<Float> operator/(const Float& n, const vec2<Float>& v);
	__duel__ const vec2<Float> operator/(const vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float> operator/(const vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float> operator/(const vec2<Int>& v0, const vec2<Float>& v1);
	__duel__ const vec2<Float> operator/(const vec2<Float>& v0, const vec2<Float>& v1);

	__duel__ const vec2<Float>& operator/=(vec2<Float>& v, const Float& n);
	__duel__ const vec2<Float>& operator/=(vec2<Float>& v0, const vec2<Int>& v1);
	__duel__ const vec2<Float>& operator/=(vec2<Float>& v0, const vec2<Float>& v1);

#pragma endregion

#pragma endregion

#pragma region vec3
	template<typename T>
	class vec3
	{
	public:
		T x, y, z;
	public:
		__duel__ vec3() :x(0.0), y(0.0), z(0.0) {}
		__duel__ vec3(const vec3<T>& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ vec3(vec3<T>&& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ vec3<T>& operator=(const vec3<T>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
		__duel__ vec3<T>& operator=(vec3<T>&& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
		__duel__ ~vec3() {}
	public:
		__duel__ vec3(const T& n) : x(n), y(n), z(n) {}
		__duel__ vec3(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}

		template<typename U>
		__duel__ vec3<T>(const vec3<U>& v) : x(v.x), y(v.y), z(v.z) {}

	public:
		__duel__ T& operator[](const T& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "The <idx> in vec3<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			default:
				CHECK(false, "Unexpectedly call this T& vec3<T>::operator[](const T& idx), unscientific!");
				break;
			}
		}
	};

	template<>
	class vec3<Int>
	{
	public:
		Int x, y, z;
	public:
		__duel__ vec3() :x(0), y(0), z(0) {}
		__duel__ vec3(const vec3<Int>& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ vec3(vec3<Int>&& v) : x(v.x), y(v.y), z(v.z) {}
		__duel__ const vec3<Int>& operator=(const vec3<Int>& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
		__duel__ const vec3<Int>& operator=(vec3<Int>&& v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
		__duel__ ~vec3() {}
	public:
		__duel__ vec3(const Int& n) : x(n), y(n), z(n) {}
		__duel__ vec3(const Int& _x, const Int& _y, const Int& _z) : x(_x), y(_y), z(_z) {}

		__duel__ vec3(const Float& n) = delete;
		__duel__ vec3(const Float& _x, const Float& _y, const Float& _z) = delete;

		__duel__ explicit vec3(const vec3<Float>& v) : x(v.x), y(v.y), z(v.z) {}

		__duel__ const vec3<Int>& operator=(const Int& n)
		{
			x = n;
			y = n;
			z = n;
			return *this;
		}
		__duel__ const vec3<Int>& operator=(Int&& n)
		{
			x = n;
			y = n;
			z = n;
			return *this;
		}

		__duel__ const vec3<Int>& operator=(const Float& n) = delete;
		__duel__ const vec3<Int>& operator=(Float&& n) = delete;
	public:
		__duel__ Int& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx <= 2, "The <idx> in vec3<T>::operator[idx] is illegal!");
			switch (idx)
			{
			case 0: return x; break;
			case 1: return y; break;
			case 2: return x; break;
			default:
				CHECK(false, "Unexpectedly call this T& vec3<T>::operator[](const T& idx), unscientific!");
				break;
			}
		}
	};

	typedef vec3<Int> vec3i;
	typedef vec3<Float> vec3f;

#pragma region vec3 dot operation

	const Int dot(const vec3<Int>& v0, const vec3<Int>& v1);
	const Float dot(const vec3<Int>& v0, const vec3<Float>& v1);
	const Float dot(const vec3<Float>& v0, const vec3<Int>& v1);
	const Float dot(const vec3<Float>& v0, const vec3<Float>& v1);

	const Int dot(const Int& n, const vec3<Int>& v);
	const Float dot(const Int& n, const vec3<Float>& v);
	const Float dot(const Float& n, const vec3<Int>& v);
	const Float dot(const Float& n, const vec3<Float>& v);

	const Int dot(const vec3<Int>& v, const Int& n);
	const Float dot(const vec3<Int>& v, const Float& n);
	const Float dot(const vec3<Float>& v, const Int& n);
	const Float dot(const vec3<Float>& v, const Float& n);

#pragma endregion

#pragma region vec3i add operation

	__duel__ const vec3<Int> operator+(const Int& n, const vec3<Int>& v);
	__duel__ const vec3<Int> operator+(const vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int> operator+(const vec3<Int>& v0, const vec3<Int>& v1);

	__duel__ const vec3<Int>& operator+=(vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int>& operator+=(vec3<Int>& v0, const vec3<Int>& v1);

#pragma endregion

#pragma region vec3i subtract operation

	__duel__ const vec3<Int> operator-(const Int& n, const vec3<Int>& v);
	__duel__ const vec3<Int> operator-(const vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int> operator-(const vec3<Int>& v0, const vec3<Int>& v1);

	__duel__ const vec3<Int>& operator-=(vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int>& operator-=(vec3<Int>& v0, const vec3<Int>& v1);

#pragma endregion

#pragma region vec3i multiply operation

	__duel__ const vec3<Int> operator*(const Int& n, const vec3<Int>& v);
	__duel__ const vec3<Int> operator*(const vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int> operator*(const vec3<Int>& v0, const vec3<Int>& v1);

	__duel__ const vec3<Int>& operator*=(vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int>& operator*=(vec3<Int>& v0, const vec3<Int>& v1);

#pragma endregion

#pragma region vec3i divide operation

	__duel__ const vec3<Int> operator/(const Int& n, const vec3<Int>& v);
	__duel__ const vec3<Int> operator/(const vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int> operator/(const vec3<Int>& v0, const vec3<Int>& v1);

	__duel__ const vec3<Int>& operator/=(vec3<Int>& v, const Int& n);
	__duel__ const vec3<Int>& operator/=(vec3<Int>& v0, const vec3<Int>& v1);

#pragma endregion

#pragma region vec3f add operation

	__duel__ const vec3<Float> operator+(const Float& n, const vec3<Float>& v);
	__duel__ const vec3<Float> operator+(const vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float> operator+(const vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float> operator+(const vec3<Int>& v0, const vec3<Float>& v1);
	__duel__ const vec3<Float> operator+(const vec3<Float>& v0, const vec3<Float>& v1);

	__duel__ const vec3<Float>& operator+=(vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float>& operator+=(vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float>& operator+=(vec3<Float>& v0, const vec3<Float>& v1);

#pragma endregion

#pragma region vec3f subtract operation

	__duel__ const vec3<Float> operator-(const Float& n, const vec3<Float>& v);
	__duel__ const vec3<Float> operator-(const vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float> operator-(const vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float> operator-(const vec3<Int>& v0, const vec3<Float>& v1);
	__duel__ const vec3<Float> operator-(const vec3<Float>& v0, const vec3<Float>& v1);

	__duel__ const vec3<Float>& operator-=(vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float>& operator-=(vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float>& operator-=(vec3<Float>& v0, const vec3<Float>& v1);

#pragma endregion

#pragma region vec3f multiply operation

	__duel__ const vec3<Float> operator*(const Float& n, const vec3<Float>& v);
	__duel__ const vec3<Float> operator*(const vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float> operator*(const vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float> operator*(const vec3<Int>& v0, const vec3<Float>& v1);
	__duel__ const vec3<Float> operator*(const vec3<Float>& v0, const vec3<Float>& v1);

	__duel__ const vec3<Float>& operator*=(vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float>& operator*=(vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float>& operator*=(vec3<Float>& v0, const vec3<Float>& v1);

#pragma endregion

#pragma region vec3f divide operation

	__duel__ const vec3<Float> operator/(const Float& n, const vec3<Float>& v);
	__duel__ const vec3<Float> operator/(const vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float> operator/(const vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float> operator/(const vec3<Int>& v0, const vec3<Float>& v1);
	__duel__ const vec3<Float> operator/(const vec3<Float>& v0, const vec3<Float>& v1);

	__duel__ const vec3<Float>& operator/=(vec3<Float>& v, const Float& n);
	__duel__ const vec3<Float>& operator/=(vec3<Float>& v0, const vec3<Int>& v1);
	__duel__ const vec3<Float>& operator/=(vec3<Float>& v0, const vec3<Float>& v1);

#pragma endregion

#pragma endregion

	class Mat4x4
	{
	public:
		Float m[4][4];
	public:
		__duel__ Mat4x4()
		{
			for (Uint i = 0; i < 4; i++)
			{
				for (Uint j = 0; j < 4; j++)
				{
					m[i][j] = i == j ? 1 : 0;
				}
			}
		}
		__duel__ Mat4x4(const Mat4x4& mat)
		{
			for (Uint i = 0; i < 4; i++)
			{
				for (Uint j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		__duel__ Mat4x4(Mat4x4&& mat)
		{
			for (Uint i = 0; i < 4; i++)
			{
				for (Uint j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		__duel__ const Mat4x4& operator=(const Mat4x4& mat)
		{
			for (Uint i = 0; i < 4; i++)
			{
				for (Uint j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		__duel__ const Mat4x4& operator=(Mat4x4&& mat)
		{
			for (Uint i = 0; i < 4; i++)
			{
				for (Uint j = 0; j < 4; j++)
				{
					m[i][j] = mat.m[i][j];
				}
			}
		}
		__duel__ ~Mat4x4() {}
	public:
		Float& operator[](const Uint& i)
		{
			CHECK(0 <= i && i < 16, "Mat4x4:: operator[](Uint idx): idx is out of range!");
			return m[(i - i % 4) / 4][i % 4];
		}


	};
}

#endif // !__CUDA3DMATH__CUH__

