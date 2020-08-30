#include "Cuda3DMath.cuh"

#pragma region vec2 dot operation

const Int dot(const vec2<Int>& v0, const vec2<Int>& v1)
{
	return v0.x*v1.x + v0.y*v1.y;
}
const Float dot(const vec2<Int>& v0, const vec2<Float>& v1)
{
	return v0.x*v1.x + v0.y*v1.y;
}
const Float dot(const vec2<Float>& v0, const vec2<Int>& v1)
{
	return v0.x*v1.x + v0.y*v1.y;
}
const Float dot(const vec2<Float>& v0, const vec2<Float>& v1)
{
	return v0.x*v1.x + v0.y*v1.y;
}

const Int dot(const Int& n, const vec2<Int>& v)
{
	return n * v.x + n * v.y;
}
const Float dot(const Int& n, const vec2<Float>& v)
{
	return n * v.x + n * v.y;
}
const Float dot(const Float& n, const vec2<Int>& v)
{
	return n * v.x + n * v.y;
}
const Float dot(const Float& n, const vec2<Float>& v)
{
	return n * v.x + n * v.y;
}

const Int dot(const vec2<Int>& v, const Int& n)
{
	return v.x*n + v.y*n;
}

const Float dot(const vec2<Int>& v, const Float& n)
{
	return v.x*n + v.y*n;
}

const Float dot(const vec2<Float>& v, const Int& n)
{
	return v.x*n + v.y*n;
}

const Float dot(const vec2<Float>& v, const Float& n)
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

__duel__ const vec2<Float> operator+(const Float& n, const vec2<Float>& v)
{
	return vec2<Float>(n + v.x, n + v.y);
}

__duel__ const vec2<Float> operator+(const vec2<Float>& v, const Float& n)
{
	return vec2<Float>(v.x + n, v.y + n);
}
__duel__ const vec2<Float> operator+(const vec2<Float>& v0, const vec2<Int>& v1)
{
	return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
}

__duel__ const vec2<Float> operator+(const vec2<Int>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
}

__duel__ const vec2<Float> operator+(const vec2<Float>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x + v1.x, v0.y + v1.y);
}

__duel__ const vec2<Float>& operator+=(vec2<Float>& v, const Float& n)
{
	v.x += n;
	v.y += n;
	return v;
}

__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<Int>& v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
	return v0;
}
__duel__ const vec2<Float>& operator+=(vec2<Float>& v0, const vec2<Float>& v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
	return v0;
}

#pragma endregion

#pragma region vec2f subtract operation

__duel__ const vec2<Float> operator-(const Float& n, const vec2<Float>& v)
{
	return vec2<Float>(n - v.x, n - v.y);
}

__duel__ const vec2<Float> operator-(const vec2<Float>& v, const Float& n)
{
	return vec2<Float>(v.x - n, v.y - n);
}
__duel__ const vec2<Float> operator-(const vec2<Float>& v0, const vec2<Int>& v1)
{
	return vec2<Float>(v0.x - v1.x, v0.y - v1.y);
}

__duel__ const vec2<Float> operator-(const vec2<Int>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x - v1.x, v0.y - v1.y);
}

__duel__ const vec2<Float> operator-(const vec2<Float>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x - v1.x, v0.y - v1.y);
}

__duel__ const vec2<Float>& operator-=(vec2<Float>& v, const Float& n)
{
	v.x -= n;
	v.y -= n;
	return v;
}

__duel__ const vec2<Float>& operator-=(vec2<Float>& v0, const vec2<Int>& v1)
{
	v0.x -= v1.x;
	v0.y -= v1.y;
	return v0;
}
__duel__ const vec2<Float>& operator-=(vec2<Float>& v0, const vec2<Float>& v1)
{
	v0.x -= v1.x;
	v0.y -= v1.y;
	return v0;
}

#pragma endregion

#pragma region vec2f multiply operation

__duel__ const vec2<Float> operator*(const Float& n, const vec2<Float>& v)
{
	return vec2<Float>(n * v.x, n * v.y);
}

__duel__ const vec2<Float> operator*(const vec2<Float>& v, const Float& n)
{
	return vec2<Float>(v.x * n, v.y * n);
}
__duel__ const vec2<Float> operator*(const vec2<Float>& v0, const vec2<Int>& v1)
{
	return vec2<Float>(v0.x * v1.x, v0.y * v1.y);
}

__duel__ const vec2<Float> operator*(const vec2<Int>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x * v1.x, v0.y * v1.y);
}

__duel__ const vec2<Float> operator*(const vec2<Float>& v0, const vec2<Float>& v1)
{
	return vec2<Float>(v0.x * v1.x, v0.y * v1.y);
}

__duel__ const vec2<Float>& operator*=(vec2<Float>& v, const Float& n)
{
	v.x *= n;
	v.y *= n;
	return v;
}

__duel__ const vec2<Float>& operator*=(vec2<Float>& v0, const vec2<Int>& v1)
{
	v0.x *= v1.x;
	v0.y *= v1.y;
	return v0;
}
__duel__ const vec2<Float>& operator*=(vec2<Float>& v0, const vec2<Float>& v1)
{
	v0.x *= v1.x;
	v0.y *= v1.y;
	return v0;
}

#pragma endregion

#pragma region vec2f divide operation

__duel__ const vec2<Float> operator/(const Float& n, const vec2<Float>& v)
{
	CHECK(v.x != 0, "(Float)n/vec2<Float>(v).x: can not devide 0!");
	CHECK(v.y != 0, "(Float)n/vec2<Float>(v).y: can not devide 0!");

	return vec2<Float>(n / v.x, n / v.y);
}

__duel__ const vec2<Float> operator/(const vec2<Float>& v, const Float& n)
{
	CHECK(n != 0, "vec2<Float>(v)/(Float)n: can not devide 0!");

	return vec2<Float>(v.x / n, v.y / n);
}
__duel__ const vec2<Float> operator/(const vec2<Float>& v0, const vec2<Int>& v1)
{
	CHECK(v1.x != 0, "vec2<Float>(v0).x/vec2<Int>(v1).x: can not devide 0!");
	CHECK(v1.y != 0, "vec2<Float>(v0).y/vec2<Int>(v1).y: can not devide 0!");

	return vec2<Float>(v0.x / v1.x, v0.y / v1.y);
}

__duel__ const vec2<Float> operator/(const vec2<Int>& v0, const vec2<Float>& v1)
{
	CHECK(v1.x != 0, "vec2<Int>(v0).x/vec2<Float>(v1).x: can not devide 0!");
	CHECK(v1.y != 0, "vec2<Int>(v0).y/vec2<Float>(v1).y: can not devide 0!");

	return vec2<Float>(v0.x / v1.x, v0.y / v1.y);
}

__duel__ const vec2<Float> operator/(const vec2<Float>& v0, const vec2<Float>& v1)
{
	CHECK(v1.x != 0, "vec2<Float>(v0).x/vec2<Float>(v1).x: can not devide 0!");
	CHECK(v1.y != 0, "vec2<Float>(v0).y/vec2<Float>(v1).y: can not devide 0!");

	return vec2<Float>(v0.x / v1.x, v0.y / v1.y);
}

__duel__ const vec2<Float>& operator/=(vec2<Float>& v, const Float& n)
{
	CHECK(n != 0, "vec2<Float>(v)/=(Float)n: can not devide 0!");

	v.x /= n;
	v.y /= n;
	return v;
}

__duel__ const vec2<Float>& operator/=(vec2<Float>& v0, const vec2<Int>& v1)
{
	CHECK(v1.x != 0, "vec2<Float>(v0).x/=vec2<Int>(v1).x: can not devide 0!");
	CHECK(v1.y != 0, "vec2<Float>(v0).y/=vec2<Int>(v1).y: can not devide 0!");

	v0.x /= v1.x;
	v0.y /= v1.y;
	return v0;
}
__duel__ const vec2<Float>& operator/=(vec2<Float>& v0, const vec2<Float>& v1)
{
	CHECK(v1.x != 0, "vec2<Float>(v0).x/=vec2<Float>(v1).x: can not devide 0!");
	CHECK(v1.y != 0, "vec2<Float>(v0).y/=vec2<Float>(v1).y: can not devide 0!");

	v0.x /= v1.x;
	v0.y /= v1.y;
	return v0;
}

#pragma endregion
