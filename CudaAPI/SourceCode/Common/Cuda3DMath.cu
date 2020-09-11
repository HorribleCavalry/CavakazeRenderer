#include "Cuda3DMath.cuh"

namespace CUM
{

#pragma region vec2

//#pragma region vec2 vector operation
//
//	template<typename T>
//	__duel__ const T dot(const vec2<T>& v0, const vec2<T>& v1)
//	{
//		return v0.x*v1.x + v0.y + v1.y;
//	}
//
//	template<typename T, typename U>
//	__duel__ Float dot(const vec2<T>& v0, const vec2<U>& v1)
//	{
//		return v0.x*v1.x + v0.y + v1.y;
//	}
//
//#pragma endregion

//#pragma region vec2 same type operation
//
//#pragma region vec2 same type operation +
//
//	template<typename T>
//	__duel__ const vec2<T> operator+(const T & n, const vec2<T>& v)
//	{
//		return vec2<T>(n+v.x, n+v.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator+(const vec2<T>& v, const T & n)
//	{
//		return vec2<T>(v.x + n, v.y + n);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator+(const vec2<T>& v0, const vec2<T>& v1)
//	{
//		return vec2<T>(v0.x + v1.x, v0.y + v1.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator+=(vec2<T>& v, const T & n)
//	{
//		v.x += n;
//		v.y += n;
//		return v;
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator+=(vec2<T>& v0, const vec2<T>& v1)
//	{
//		v0.x += v1.x;
//		v0.y += v1.y;
//		return v0;
//	}
//
//#pragma endregion
//
//#pragma region vec2 same type operation -
//
//	template<typename T>
//	__duel__ const vec2<T> operator-(const T & n, const vec2<T>& v)
//	{
//		return vec2<T>(n - v.x, n - v.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator-(const vec2<T>& v, const T & n)
//	{
//		return vec2<T>(v.x - n, v.y - n);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator-(const vec2<T>& v0, const vec2<T>& v1)
//	{
//		return vec2<T>(v0.x - v1.x, v0.y - v1.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator-=(vec2<T>& v, const T & n)
//	{
//		v.x -= n;
//		v.y -= n;
//		return v;
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator-=(vec2<T>& v0, const vec2<T>& v1)
//	{
//		v0.x -= v1.x;
//		v0.y -= v1.y;
//		return v0;
//	}
//
//#pragma endregion
//
//#pragma region vec2 same type operation *
//	template<typename T>
//	__duel__ const vec2<T> operator*(const T & n, const vec2<T>& v)
//	{
//		return vec2<T>(n * v.x, n * v.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator*(const vec2<T>& v, const T & n)
//	{
//		return vec2<T>(v.x * n, v.y * n);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator*(const vec2<T>& v0, const vec2<T>& v1)
//	{
//		return vec2<T>(v0.x * v1.x, v0.y * v1.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator*=(vec2<T>& v, const T & n)
//	{
//		v.x *= n;
//		v.y *= n;
//		return v;
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator*=(vec2<T>& v0, const vec2<T>& v1)
//	{
//		v0.x *= v1.x;
//		v0.y *= v1.y;
//		return v0;
//	}
//
//#pragma endregion
//
//#pragma region vec2 same type operation /
//	template<typename T>
//	__duel__ const vec2<T> operator/(const T & n, const vec2<T>& v)
//	{
//		CHECK(v.x != 0, "Vec2 with sampe type divide error in n/v: v.x can not be 0!");
//		CHECK(v.y != 0, "Vec2 with sampe type divide error in n/v: v.y can not be 0!");
//		return vec2<T>(n / v.x, n / v.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator/(const vec2<T>& v, const T & n)
//	{
//		CHECK(n != 0, "Vec2 with sampe type divide error in v/n: n can not be 0!");
//		return vec2<T>(v.x / n, v.y / n);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T> operator/(const vec2<T>& v0, const vec2<T>& v1)
//	{
//		CHECK(v1.x != 0, "Vec2 with sampe type divide error in v0/v1: v1.x can not be 0!");
//		CHECK(v1.y != 0, "Vec2 with sampe type divide error in v0/v1: v1.y can not be 0!");
//		return vec2<T>(v0.x / v1.x, v0.y / v1.y);
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator/=(vec2<T>& v, const T & n)
//	{
//		CHECK(n != 0, "Vec2 with sampe type divide error in v/=n: n can not be 0!");
//		v.x /= n;
//		v.y /= n;
//		return v;
//	}
//
//	template<typename T>
//	__duel__ const vec2<T>& operator/=(vec2<T>& v0, const vec2<T>& v1)
//	{
//		CHECK(v1.x != 0, "Vec2 with sampe type divide error in v0/=v1: v1.x can not be 0!");
//		CHECK(v1.y != 0, "Vec2 with sampe type divide error in v0/=v1: v1.y can not be 0!");
//		v0.x /= v1.x;
//		v0.y /= v1.y;
//		return v0;
//	}
//
//#pragma endregion
//
//#pragma endregion

#pragma region vec2 different type operation

//#pragma region vec2 different type operation +
//
//	template<typename T, typename U>
//	__duel__ const vec2<Float> operator+(const T & n, const vec2<U>& v)
//	{
//		return vec2<Float>(n + v.x, n + v.y);
//	}
//
//	template<typename T, typename U>
//	__duel__ const vec2<Float> operator+(const vec2<T>& v, const U & n)
//	{
//		return vec2<Float>(v.x + n, v.y + n);
//	}
//
//	template<typename T, typename U>
//	__duel__ const vec2<Float> operator+(const vec2<T>& v0, const vec2<T>& v1)
//	{
//		return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
//	}
//
//	__duel__ const vec2<Float>& operator+=(vec2<Float>& v, const Int & n)
//	{
//		v.x += n;
//		v.y += n;
//		return v;
//	}
//
//	__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<Int>& v1)
//	{
//		v0.x += v1.x;
//		v0.y += v1.y;
//		return v0;
//	}
//
//#pragma endregion


#pragma endregion



#pragma endregion




	#pragma region mat4x4
	//__device__  const Mat4x4 Mat4x4::identiy(
	//	1, 0, 0, 0,
	//	0, 1, 0, 0,
	//	0, 0, 1, 0,
	//	0, 0, 0, 1);

	//template<typename T>
	//__duel__ const Mat4x4<T> operator+(const Mat4x4<T>& mat0, const Mat4x4<T>& mat1)
	//{
	//	Mat4x4<T> result;
	//	for (Uint i = 0; i < 4; i++)
	//	{
	//		for (Uint j = 0; j < 4; j++)
	//		{
	//			result.m[i][j] = mat0.m[i][j] + mat1.m[i][j];
	//		}
	//	}
	//	return result;
	//}
	#pragma endregion
}