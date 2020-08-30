﻿#include "Cuda3DMath.cuh"

#pragma region vec2 dot operation

auto dot(const vec2<Int>& v0, const vec2<Int>& v1) ->decltype(v0.x*v1.x)
{
	return v0.x*v1.x + v0.y*v1.y;
}
auto dot(const vec2<Int>& v0, const vec2<Float>& v1) ->decltype(v0.x*v1.x)
{
	return v0.x*v1.x + v0.y*v1.y;
}
auto dot(const vec2<Float>& v0, const vec2<Int>& v1) ->decltype(v0.x*v1.x)
{
	return v0.x*v1.x + v0.y*v1.y;
}
auto dot(const vec2<Float>& v0, const vec2<Float>& v1) ->decltype(v0.x*v1.x)
{
	return v0.x*v1.x + v0.y*v1.y;
}

auto dot(const Int& n, const vec2<Int>& v) ->decltype(n*v.x)
{
	return n * v.x + n * v.y;
}
auto dot(const Int& n, const vec2<Float>& v) ->decltype(n*v.x)
{
	return n * v.x + n * v.y;
}
auto dot(const Float& n, const vec2<Int>& v) ->decltype(n*v.x)
{
	return n * v.x + n * v.y;
}
auto dot(const Float& n, const vec2<Float>& v) ->decltype(n*v.x)
{
	return n * v.x + n * v.y;
}

auto dot(const vec2<Int>& v, const Int& n) -> decltype(v.x*n)
{
	return v.x*n + v.y*n;
}
auto dot(const vec2<Int>& v, const Float& n) -> decltype(v.x*n)
{
	return v.x*n + v.y*n;
}
auto dot(const vec2<Float>& v, const Int& n) -> decltype(v.x*n)
{
	return v.x*n + v.y*n;
}
auto dot(const vec2<Float>& v, const Float& n) -> decltype(v.x*n)
{
	return v.x*n + v.y*n;
}

#pragma endregion


#pragma region vec2i add operation

__duel__ const vec2<Int> operator+(const Int & n, const vec2<Int>& v)
{
	return vec2<Int>(n + v.x, n + v.y);
}

__duel__ const vec2<Int> operator+(const vec2<Int>& v, const Int & n)
{
	return vec2<Int>(v.x + n, v.y + n);
}

__duel__ const vec2<Int> operator+(const vec2<Int>& v0, const vec2<Int>& v1)
{
	return vec2<Int>(v0.x + v1.x, v0.y + v1.y);
}

__duel__ const vec2<Int>& operator+=(vec2<Int>& v, const Int & n)
{
	v.x += n;
	v.y += n;
	return v;
}

__duel__ const vec2<Int>& operator+=(vec2<Int>& v0, const vec2<Int>& v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
	return v0;
}

#pragma endregion

#pragma region vec2i subtract operation

__duel__ const vec2<Int> operator-(const Int & n, const vec2<Int>& v)
{
	return vec2<Int>(n - v.x, n - v.y);
}

__duel__ const vec2<Int> operator-(const vec2<Int>& v, const Int & n)
{
	return vec2<Int>(v.x - n, v.y - n);
}

__duel__ const vec2<Int> operator-(const vec2<Int>& v0, const vec2<Int>& v1)
{
	return vec2<Int>(v0.x - v1.x, v0.y - v1.y);
}

__duel__ const vec2<Int>& operator-=(vec2<Int>& v, const Int & n)
{
	v.x -= n;
	v.y -= n;
	return v;
}

__duel__ const vec2<Int>& operator-=(vec2<Int>& v0, const vec2<Int>& v1)
{
	v0.x -= v1.x;
	v0.y -= v1.y;
	return v0;
}

#pragma endregion

#pragma region vec2i multiply operation

__duel__ const vec2<Int> operator*(const Int & n, const vec2<Int>& v)
{
	return vec2<Int>(n * v.x, n * v.y);
}

__duel__ const vec2<Int> operator*(const vec2<Int>& v, const Int & n)
{
	return vec2<Int>(v.x * n, v.y * n);
}

__duel__ const vec2<Int> operator*(const vec2<Int>& v0, const vec2<Int>& v1)
{
	return vec2<Int>(v0.x * v1.x, v0.y * v1.y);
}

__duel__ const vec2<Int>& operator*=(vec2<Int>& v, const Int & n)
{
	v.x *= n;
	v.y *= n;
	return v;
}

__duel__ const vec2<Int>& operator*=(vec2<Int>& v0, const vec2<Int>& v1)
{
	v0.x *= v1.x;
	v0.y *= v1.y;
	return v0;
}

#pragma endregion

#pragma region vec2i divide operation

__duel__ const vec2<Int> operator/(const Int & n, const vec2<Int>& v)
{
	CHECK(v.x != 0, "(Int)n/vec2<int>(v).x: can not divide 0!");
	CHECK(v.y != 0, "(Int)n/vec2<int>(v).y: can not divide 0!");

	return vec2<Int>(n / v.x, n / v.y);
}

__duel__ const vec2<Int> operator/(const vec2<Int>& v, const Int & n)
{
	CHECK(n != 0, "vec2<int>(v)/n: can not divide 0!");

	return vec2<Int>(v.x / n, v.y / n);
}

__duel__ const vec2<Int> operator/(const vec2<Int>& v0, const vec2<Int>& v1)
{
	CHECK(v1.x != 0, "vec2<int>(v0).x/vec2<int>(v1).x: can not divide 0!");
	CHECK(v1.y != 0, "vec2<int>(v0).y/vec2<int>(v1).y: can not divide 0!");

	return vec2<Int>(v0.x / v1.x, v0.y / v1.y);
}

__duel__ const vec2<Int>& operator/=(vec2<Int>& v, const Int & n)
{
	CHECK(n != 0, "vec2<int>(v).y/=(int)n: can not divide 0!");

	v.x /= n;
	v.y /= n;
	return v;
}

__duel__ const vec2<Int>& operator/=(vec2<Int>& v0, const vec2<Int>& v1)
{
	CHECK(v1.x != 0, "vec2<int>(v0).x/=vec2<int>(v1).x: can not divide 0!");
	CHECK(v1.y != 0, "vec2<int>(v0).y/=vec2<int>(v1).y: can not divide 0!");

	v0.x /= v1.x;
	v0.y /= v1.y;
	return v0;
}

#pragma endregion


#pragma region vec2f add operation
template<typename T>
__duel__ const vec2<Float>& operator+(const T & n, const vec2<Float>& v)
{
	return vec2<Float>(n + v.x, n + v.y);
}
template<typename T>
__duel__ const vec2<Float>& operator+(const vec2<Float>& v, const T & n)
{
	return vec2<Float>(v.x + n, v.y + n);
}
template<typename T>
__duel__ const vec2<Float>& operator+(const vec2<Float>& v0, const vec2<T>& v1)
{
	return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator+(const vec2<T>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator+=(vec2<Float>& v, const T & n)
{
	v.x += n;
	v.y += n;
	return v;
}
template<typename T>
__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<T>& v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
	return v0;
}
#pragma endregion

#pragma region vec2f subtract operation
template<typename T>
__duel__ const vec2<Float>& operator-(const T & n, const vec2<Float>& v)
{
	return vec2<Float>(n - v.x, n - v.y);
}
template<typename T>
__duel__ const vec2<Float>& operator-(const vec2<Float>& v, const T & n)
{
	return vec2<Float>(v.x - n, v.y - n);
}
template<typename T>
__duel__ const vec2<Float>& operator-(const vec2<Float>& v0, const vec2<T>& v1)
{
	return vec2<Float>(v0.x - v1.x, v0.y - v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator-(const vec2<T>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x - v1.x, v0.y - v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator-=(vec2<Float>& v, const T & n)
{
	v.x -= n;
	v.y -= n;
	return v;
}
template<typename T>
__duel__ const vec2<Float>& operator-=(vec2<Float>& v0, const vec2<T>& v1)
{
	v0.x -= v1.x;
	v0.y -= v1.y;
	return v0;
}
#pragma endregion

#pragma region vec2f multiply operation
template<typename T>
__duel__ const vec2<Float>& operator*(const T & n, const vec2<Float>& v)
{
	return vec2<Float>(n * v.x, n * v.y);
}
template<typename T>
__duel__ const vec2<Float>& operator*(const vec2<Float>& v, const T & n)
{
	return vec2<Float>(v.x * n, v.y * n);
}
template<typename T>
__duel__ const vec2<Float>& operator*(const vec2<Float>& v0, const vec2<T>& v1)
{
	return vec2<Float>(v0.x * v1.x, v0.y * v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator*(const vec2<T>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x * v1.x, v0.y * v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator*=(vec2<Float>& v, const T & n)
{
	v.x *= n;
	v.y *= n;
	return v;
}
template<typename T>
__duel__ const vec2<Float>& operator*=(vec2<Float>& v0, const vec2<T>& v1)
{
	v0.x *= v1.x;
	v0.y *= v1.y;
	return v0;
}
#pragma endregion

#pragma region vec2f multiply operation
template<typename T>
__duel__ const vec2<Float>& operator/(const T & n, const vec2<Float>& v)
{
	CHECK(v.x != 0, "(T)n/vec2<Float>(v).x: can not divide 0!");
	CHECK(v.y != 0, "(T)n/vec2<Float>(v).y: can not divide 0!");

	return vec2<Float>(n / v.x, n / v.y);
}
template<typename T>
__duel__ const vec2<Float>& operator/(const vec2<Float>& v, const T & n)
{
	CHECK(n != 0, "vec2<Float>(v)/(T)n: can not divide 0!");

	return vec2<Float>(v.x / n, v.y / n);
}
template<typename T>
__duel__ const vec2<Float>& operator/(const vec2<Float>& v0, const vec2<T>& v1)
{
	CHECK(v1.x != 0, "vec2<Float>(v0).x/vec2<T>(v1).x: can not divide 0!");
	CHECK(v1.y != 0, "vec2<Float>(v0).y/vec2<T>(v1).y: can not divide 0!");

	return vec2<Float>(v0.x / v1.x, v0.y / v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator/(const vec2<T>& v0, const vec2<Float>& v1)
{
	CHECK(v1.x != 0, "vec2<T>(v0).x/vec2<Float>(v1).x: can not divide 0!");
	CHECK(v1.y != 0, "vec2<T>(v0).y/vec2<Float>(v1).y: can not divide 0!");

	return vec2<Float>(v0.x / v1.x, v0.y / v1.y);
}
template<typename T>
__duel__ const vec2<Float>& operator/=(vec2<Float>& v, const T & n)
{
	CHECK(n != 0, "vec2<Float>(v)/=(T)n: can not divide 0!");

	v.x /= n;
	v.y /= n;
	return v;
}
template<typename T>
__duel__ const vec2<Float>& operator/=(vec2<Float>& v0, const vec2<T>& v1)
{
	CHECK(v1.x != 0, "vec2<Float>(v0).x/=vec2<T>(v1).x: can not divide 0!");
	CHECK(v1.y != 0, "vec2<Float>(v0).y/=vec2<T>(v1).y: can not divide 0!");

	v0.x /= v1.x;
	v0.y /= v1.y;
	return v0;
}
#pragma endregion