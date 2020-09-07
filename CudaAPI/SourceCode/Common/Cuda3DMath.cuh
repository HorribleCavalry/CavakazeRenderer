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
		__duel__ vec2() :x(0.0), y(0.0) {}
		__duel__ vec2(const vec2<T>& v) : x(v.x), y(v.y) {}
		__duel__ ~vec2() {}
	public:

		__duel__ const vec2& operator=(const vec2<int>& v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}

	public:
		__duel__ T& operator[](const T& idx)
		{
			CHECK(idx >= 0 && idx <= 1, "The <idx> in vec2<T>::operator[idx] is illegal!");
			return idx == 0 ? x : y;
		}
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

	template<typename T>
	__duel__ void logData(const vec2<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << custd::endl;
	}

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

	template<typename T>
	__duel__ void logData(const vec3<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << "\t" << v.z << custd::endl;
	}

	template<typename T>
	class vec4
	{
	public:
		T x, y, z, w;
	public:
		__duel__ vec4(T _x, T _y, T _z, T _w)
			:x(_x), y(_y), z(_z), w(_w)
		{
		}
	public:
		__duel__ vec4() : vec4(0, 0, 0, 0) {}
		__duel__ vec4(const vec4& v) : vec4(v.x, v.y, v.z, v.w) {}
		__duel__ vec4(vec4&& v) : vec4(v) {}
	};
	typedef vec4<Int> vec4i;
	typedef vec4<Float> vec4f;

	template<typename T>
	__duel__ void logData(const vec4<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << "\t" << v.z << "\t" << v.w << custd::endl;
	}

	template<typename T>
	class Mat4x4
	{
	public:
		T m[4][4];
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

	public:
		__duel__ Mat4x4(
			T m00, T m01, T m02, T m03,
			T m10, T m11, T m12, T m13,
			T m20, T m21, T m22, T m23,
			T m30, T m31, T m32, T m33
		)
			//:
			//m[0][0](m00), m[0][1](m01), m[0][2](m02), m[0][3](m03),
			//m[1][0](m10), m[1][1](m11), m[1][2](m12), m[1][3](m13),
			//m[2][0](m20), m[2][1](m21), m[2][2](m22), m[2][3](m23),
			//m[3][0](m30), m[3][1](m31), m[3][2](m32), m[3][3](m33)
		{
			m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
			m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
			m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
			m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
		}
	public:
		__duel__ vec4<T>&& Row(Uint idx) const
		{
			CHECK(idx >= 0 && idx <= 3, "Mat4x4::Row(idx): idx is out of range!");
			switch (idx)
			{
			case 0: return vec4<T>(m[0][0], m[0][1], m[0][2], m[0][3]); break;
			case 1: return vec4<T>(m[1][0], m[1][1], m[1][2], m[1][3]); break;
			case 2: return vec4<T>(m[2][0], m[2][1], m[2][2], m[2][3]); break;
			case 3: return vec4<T>(m[3][0], m[3][1], m[3][2], m[3][3]); break;
			default:
				CHECK(false, "It can not run Mat4x4::Row(), called switch::default");
				break;
			}
		}

		__duel__ vec4<T>&& Column(Uint idx) const
		{
			CHECK(idx >= 0 && idx <= 3, "Mat4x4::Column(idx): idx is out of range!");
			switch (idx)
			{
			case 0: return vec4<T>(m[0][0], m[1][0], m[2][0], m[3][0]); break;
			case 1: return vec4<T>(m[0][1], m[1][1], m[2][1], m[3][1]); break;
			case 2: return vec4<T>(m[0][2], m[1][2], m[2][2], m[3][2]); break;
			case 3: return vec4<T>(m[0][3], m[1][3], m[2][3], m[3][3]); break;
			default:
				CHECK(false, "It can not run Mat4x4::Column(), called switch::default");
				break;
			}
		}

		__duel__ Mat4x4& transpose()
		{
			Float temp;
			for (Uint i = 0; i < 4; i++)
			{
				for (Uint j = i + 1; j < 4; i++)
				{
					temp = m[i][j];
					m[i][j] = m[j][i];
					m[j][i] = temp;
				}
			}

			return *this;
		}

	public:


	public:
		//__device__  static const Mat4x4 identiy;
	};

	template<typename T>
	__duel__ const Mat4x4<T> operator+(const Mat4x4<T>& mat0, const Mat4x4<T>& mat1)
	{
		Mat4x4<T> result;
		for (Uint i = 0; i < 4; i++)
		{
			for (Uint j = 0; j < 4; j++)
			{
				result.m[i][j] = mat0.m[i][j] + mat1.m[i][j];
			}
		}
		return result;
	}

	template<typename T, typename U>
	__duel__ const Mat4x4<Float> operator+(const Mat4x4<T>& mat0, const Mat4x4<U>& mat1)
	{
		Mat4x4<Float> result;
		for (Uint i = 0; i < 4; i++)
		{
			for (Uint j = 0; j < 4; j++)
			{
				result.m[i][j] = mat0.m[i][j] + mat1.m[i][j];
			}
		}
		return result;
	}

	template<typename T>
	__duel__ void logData(const Mat4x4<T>& mat)
	{
		const custd::Ostream os;
		for (Uint i = 0; i < 4; i++)
		{
			for (Uint j = 0; j < 4; j++)
			{
				os << mat.m[i][j] << "\t";
			}
			os << custd::endl;
		}
	}

}



#pragma region mat4x4 marco
#define Mat4x4_identity Mat4x4<Float>(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
//#define Mat4x4_identity Mat4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
#pragma endregion

#endif // !__CUDA3DMATH__CUH__

