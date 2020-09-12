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
		__duel__ T& operator[](const T& idx)
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
		return v0.x*v1.x + v0.y + v1.y;
	}

	template<typename T, typename U>
	__duel__ Float dot(const vec2<T>& v0, const vec2<U>& v1)
	{
		return v0.x*v1.x + v0.y + v1.y;
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

	//__duel__ const vec2<Float>& operator+=(vec2<Float>& v, const Int& n)
	//{
	//	v.x += n;
	//	v.y += n;
	//	return v;
	//}
	//__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<Int>& v1)
	//{
	//	v0.x += v1.x;
	//	v0.y += v1.y;
	//	return v0;
	//}

#pragma endregion


#pragma endregion

	template<typename T>
	__duel__ void logData(const vec2<T>& v)
	{
		const custd::Ostream os;
		os << v.x << "\t" << v.y << custd::endl;
	}

#pragma endregion

//#pragma region vec3
//	template<typename T>
//	class vec3
//	{
//	public:
//		T x, y;
//	public:
//		__duel__ vec3() :x(0.0), y(0.0) {}
//		__duel__ vec3(const T& _x, const T& _y) : x(_x), y(_y) {}
//		__duel__ vec3(const vec3<T>& v) : x(v.x), y(v.y) {}
//		template<typename U>
//		__duel__ explicit vec3(const vec3<U>& v) : x(v.x), y(v.y) {}
//		__duel__ ~vec3() {}
//	public:
//		__duel__ const vec3& operator=(const vec3<int>& v)
//		{
//			x = v.x;
//			y = v.y;
//			return *this;
//		}
//
//	public:
//		__duel__ T& operator[](const T& idx)
//		{
//			CHECK(idx >= 0 && idx <= 1, "The <idx> in vec3<T>::operator[idx] is illegal!");
//			return idx == 0 ? x : y;
//		}
//	};
//
//	typedef vec3<Int> vec3i;
//	typedef vec3<Float> vec3f;
//
//#pragma region vec3 vector operation
//
//	template<typename T>
//	__duel__ const T dot(const vec3<T>& v0, const vec3<T>& v1);
//
//	template<typename T, typename U>
//	__duel__ Float dot(const vec3<T>& v0, const vec3<U>& v1);
//
//#pragma endregion
//
//#pragma region vec3 same type operation
//
//#pragma region vec3 same type operation +
//
//	template<typename T>
//	__duel__ const vec3<T> operator+(const T& n, const vec3<T>& v);
//	template<typename T>
//	__duel__ const vec3<T> operator+(const vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T> operator+(const vec3<T>& v0, const vec3<T>& v1);
//
//	template<typename T>
//	__duel__ const vec3<T>& operator+=(vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T>& operator+=(vec3<T>& v0, const vec3<T>& v1);
//
//#pragma endregion
//
//#pragma region vec3 same type operation -
//	template<typename T>
//	__duel__ const vec3<T> operator-(const T& n, const vec3<T>& v);
//	template<typename T>
//	__duel__ const vec3<T> operator-(const vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T> operator-(const vec3<T>& v0, const vec3<T>& v1);
//
//	template<typename T>
//	__duel__ const vec3<T>& operator-=(vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T>& operator-=(vec3<T>& v0, const vec3<T>& v1);
//
//#pragma endregion
//
//#pragma region vec3 same type operation *
//
//	template<typename T>
//	__duel__ const vec3<T> operator*(const T& n, const vec3<T>& v);
//	template<typename T>
//	__duel__ const vec3<T> operator*(const vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T> operator*(const vec3<T>& v0, const vec3<T>& v1);
//
//	template<typename T>
//	__duel__ const vec3<T>& operator*=(vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T>& operator*=(vec3<T>& v0, const vec3<T>& v1);
//
//#pragma endregion
//
//#pragma region vec3 same type operation /
//
//	template<typename T>
//	__duel__ const vec3<T> operator/(const T& n, const vec3<T>& v);
//	template<typename T>
//	__duel__ const vec3<T> operator/(const vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T> operator/(const vec3<T>& v0, const vec3<T>& v1);
//
//	template<typename T>
//	__duel__ const vec3<T>& operator/=(vec3<T>& v, const T& n);
//	template<typename T>
//	__duel__ const vec3<T>& operator/=(vec3<T>& v0, const vec3<T>& v1);
//
//#pragma endregion
//
//#pragma endregion
//
//#pragma region vec3 different type operation
//
//#pragma region vec3 different type operation +
//
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator+(const T& n, const vec3<U>& v);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator+(const vec3<T>& v, const U& n);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator+(const vec3<T>& v0, const vec3<U>& v1);
//
//	__duel__ const vec3<Float>& operator+=(vec3<Float>& v, const Int& n);
//	__duel__ const vec3<Float>& operator+=(vec3<Float>& v0, const vec3<Int>& v1);
//
//#pragma endregion
//
//#pragma region vec3 different type operation -
//
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator-(const T& n, const vec3<U>& v);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator-(const vec3<T>& v, const U& n);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator-(const vec3<T>& v0, const vec3<U>& v1);
//
//	__duel__ const vec3<Float>& operator-=(vec3<Float>& v, const Int& n);
//	__duel__ const vec3<Float>& operator-=(vec3<Float>& v0, const vec3<Int>& v1);
//
//#pragma endregion
//
//#pragma region vec3 different type operation *
//
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator*(const T& n, const vec3<U>& v);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator*(const vec3<T>& v, const U& n);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator*(const vec3<T>& v0, const vec3<U>& v1);
//
//	__duel__ const vec3<Float>& operator*=(vec3<Float>& v, const Int& n);
//	__duel__ const vec3<Float>& operator*=(vec3<Float>& v0, const vec3<Int>& v1);
//
//#pragma endregion
//
//#pragma region vec3 different type operation /
//
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator/(const T& n, const vec3<U>& v);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator/(const vec3<T>& v, const U& n);
//	template<typename T, typename U>
//	__duel__ const vec3<Float> operator/(const vec3<T>& v0, const vec3<U>& v1);
//
//	__duel__ const vec3<Float>& operator/=(vec3<Float>& v, const Int& n);
//	__duel__ const vec3<Float>& operator/=(vec3<Float>& v0, const vec3<Int>& v1);
//
//#pragma endregion
//
//
//#pragma endregion
//
//	template<typename T>
//	__duel__ void logData(const vec3<T>& v)
//	{
//		const custd::Ostream os;
//		os << v.x << "\t" << v.y << custd::endl;
//	}
//
//#pragma endregion


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
		__duel__ vec4<T>&& Row(Int idx) const
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

		__duel__ vec4<T>&& Column(Int idx) const
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

