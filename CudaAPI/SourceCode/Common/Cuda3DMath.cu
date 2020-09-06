#include "Cuda3DMath.cuh"

namespace CUM
{
#pragma region vec2

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
		CHECK(v.x != 0, "(Int)n/vec2<Int>(v).x: can not divide 0!");
		CHECK(v.y != 0, "(Int)n/vec2<Int>(v).y: can not divide 0!");

		return vec2<Int>(n / v.x, n / v.y);
	}

	__duel__ const vec2<Int> operator/(const vec2<Int>& v, const Int & n)
	{
		CHECK(n != 0, "vec2<Int>(v)/n: can not divide 0!");

		return vec2<Int>(v.x / n, v.y / n);
	}

	__duel__ const vec2<Int> operator/(const vec2<Int>& v0, const vec2<Int>& v1)
	{
		CHECK(v1.x != 0, "vec2<Int>(v0).x/vec2<Int>(v1).x: can not divide 0!");
		CHECK(v1.y != 0, "vec2<Int>(v0).y/vec2<Int>(v1).y: can not divide 0!");

		return vec2<Int>(v0.x / v1.x, v0.y / v1.y);
	}

	__duel__ const vec2<Int>& operator/=(vec2<Int>& v, const Int & n)
	{
		CHECK(n != 0, "vec2<Int>(v).y/=(Int)n: can not divide 0!");

		v.x /= n;
		v.y /= n;
		return v;
	}

	__duel__ const vec2<Int>& operator/=(vec2<Int>& v0, const vec2<Int>& v1)
	{
		CHECK(v1.x != 0, "vec2<Int>(v0).x/=vec2<Int>(v1).x: can not divide 0!");
		CHECK(v1.y != 0, "vec2<Int>(v0).y/=vec2<Int>(v1).y: can not divide 0!");

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

#pragma endregion

#pragma region vec3

#pragma region vec3 dot operation

	const Int dot(const vec3<Int>& v0, const vec3<Int>& v1)
	{
		return v0.x*v1.x + v0.y*v1.y;
	}
	const Float dot(const vec3<Int>& v0, const vec3<Float>& v1)
	{
		return v0.x*v1.x + v0.y*v1.y;
	}
	const Float dot(const vec3<Float>& v0, const vec3<Int>& v1)
	{
		return v0.x*v1.x + v0.y*v1.y;
	}
	const Float dot(const vec3<Float>& v0, const vec3<Float>& v1)
	{
		return v0.x*v1.x + v0.y*v1.y;
	}

	const Int dot(const Int& n, const vec3<Int>& v)
	{
		return n * v.x + n * v.y;
	}
	const Float dot(const Int& n, const vec3<Float>& v)
	{
		return n * v.x + n * v.y;
	}
	const Float dot(const Float& n, const vec3<Int>& v)
	{
		return n * v.x + n * v.y;
	}
	const Float dot(const Float& n, const vec3<Float>& v)
	{
		return n * v.x + n * v.y;
	}

	const Int dot(const vec3<Int>& v, const Int& n)
	{
		return v.x*n + v.y*n;
	}

	const Float dot(const vec3<Int>& v, const Float& n)
	{
		return v.x*n + v.y*n;
	}

	const Float dot(const vec3<Float>& v, const Int& n)
	{
		return v.x*n + v.y*n;
	}

	const Float dot(const vec3<Float>& v, const Float& n)
	{
		return v.x*n + v.y*n;
	}

#pragma endregion

#pragma region vec3i add operation

	__duel__ const vec3<Int> operator+(const Int & n, const vec3<Int>& v)
	{
		return vec3<Int>(n + v.x, n + v.y, n+v.z);
	}

	__duel__ const vec3<Int> operator+(const vec3<Int>& v, const Int & n)
	{
		return vec3<Int>(v.x + n, v.y + n, v.z + n);
	}

	__duel__ const vec3<Int> operator+(const vec3<Int>& v0, const vec3<Int>& v1)
	{
		return vec3<Int>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	__duel__ const vec3<Int>& operator+=(vec3<Int>& v, const Int & n)
	{
		v.x += n;
		v.y += n;
		return v;
	}

	__duel__ const vec3<Int>& operator+=(vec3<Int>& v0, const vec3<Int>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		return v0;
	}

#pragma endregion

#pragma region vec3i subtract operation

	__duel__ const vec3<Int> operator-(const Int & n, const vec3<Int>& v)
	{
		return vec3<Int>(n - v.x, n - v.y, n - v.z);
	}

	__duel__ const vec3<Int> operator-(const vec3<Int>& v, const Int & n)
	{
		return vec3<Int>(v.x - n, v.y - n, v.z - n);
	}

	__duel__ const vec3<Int> operator-(const vec3<Int>& v0, const vec3<Int>& v1)
	{
		return vec3<Int>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

	__duel__ const vec3<Int>& operator-=(vec3<Int>& v, const Int & n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		return v;
	}

	__duel__ const vec3<Int>& operator-=(vec3<Int>& v0, const vec3<Int>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		return v0;
	}

#pragma endregion

#pragma region vec3i multiply operation

	__duel__ const vec3<Int> operator*(const Int & n, const vec3<Int>& v)
	{
		return vec3<Int>(n * v.x, n * v.y, n * v.z);
	}

	__duel__ const vec3<Int> operator*(const vec3<Int>& v, const Int & n)
	{
		return vec3<Int>(v.x * n, v.y * n, v.z * n);
	}

	__duel__ const vec3<Int> operator*(const vec3<Int>& v0, const vec3<Int>& v1)
	{
		return vec3<Int>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	__duel__ const vec3<Int>& operator*=(vec3<Int>& v, const Int & n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		return v;
	}

	__duel__ const vec3<Int>& operator*=(vec3<Int>& v0, const vec3<Int>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		return v0;
	}

#pragma endregion

#pragma region vec3i divide operation

	__duel__ const vec3<Int> operator/(const Int & n, const vec3<Int>& v)
	{
		CHECK(v.x != 0, "(Int)n/vec3<Int>(v).x: can not divide 0!");
		CHECK(v.y != 0, "(Int)n/vec3<Int>(v).y: can not divide 0!");
		CHECK(v.z != 0, "(Int)n/vec3<Int>(v).z: can not divide 0!");

		return vec3<Int>(n / v.x, n / v.y, n/v.z);
	}

	__duel__ const vec3<Int> operator/(const vec3<Int>& v, const Int & n)
	{
		CHECK(n != 0, "vec3<Int>(v)/n: can not divide 0!");

		return vec3<Int>(v.x / n, v.y / n, v.z / n);
	}

	__duel__ const vec3<Int> operator/(const vec3<Int>& v0, const vec3<Int>& v1)
	{
		CHECK(v1.x != 0, "vec3<Int>(v0).x/vec3<Int>(v1).x: can not divide 0!");
		CHECK(v1.y != 0, "vec3<Int>(v0).y/vec3<Int>(v1).y: can not divide 0!");
		CHECK(v1.z != 0, "vec3<Int>(v0).y/vec3<Int>(v1).z: can not divide 0!");

		return vec3<Int>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	__duel__ const vec3<Int>& operator/=(vec3<Int>& v, const Int & n)
	{
		CHECK(n != 0, "vec3<Int>(v)/=(Int)n: can not divide 0!");

		v.x /= n;
		v.y /= n;
		v.z /= n;
		return v;
	}

	__duel__ const vec3<Int>& operator/=(vec3<Int>& v0, const vec3<Int>& v1)
	{
		CHECK(v1.x != 0, "vec3<Int>(v0).x/=vec3<Int>(v1).x: can not divide 0!");
		CHECK(v1.y != 0, "vec3<Int>(v0).y/=vec3<Int>(v1).y: can not divide 0!");
		CHECK(v1.z != 0, "vec3<Int>(v0).z/=vec3<Int>(v1).z: can not divide 0!");

		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}

#pragma endregion


#pragma region vec3f add operation

	__duel__ const vec3<Float> operator+(const Float& n, const vec3<Float>& v)
	{
		return vec3<Float>(n + v.x, n + v.y, n + v.z);
	}

	__duel__ const vec3<Float> operator+(const vec3<Float>& v, const Float& n)
	{
		return vec3<Float>(v.x + n, v.y + n, v.z + n);
	}
	__duel__ const vec3<Float> operator+(const vec3<Float>& v0, const vec3<Int>& v1)
	{
		return vec3<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	__duel__ const vec3<Float> operator+(const vec3<Int>& v0, const vec3<Float>& v1)
	{
		return vec3<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	__duel__ const vec3<Float> operator+(const vec3<Float>& v0, const vec3<Float>& v1)
	{
		return vec3<Float>(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
	}

	__duel__ const vec3<Float>& operator+=(vec3<Float>& v, const Float& n)
	{
		v.x += n;
		v.y += n;
		v.z += n;
		return v;
	}

	__duel__ const vec3<Float>& operator+=(vec3<Float>& v0, const vec3<Int>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		return v0;
	}
	__duel__ const vec3<Float>& operator+=(vec3<Float>& v0, const vec3<Float>& v1)
	{
		v0.x += v1.x;
		v0.y += v1.y;
		v0.z += v1.z;
		return v0;
	}

#pragma endregion

#pragma region vec3f subtract operation

	__duel__ const vec3<Float> operator-(const Float& n, const vec3<Float>& v)
	{
		return vec3<Float>(n - v.x, n - v.y, n - v.z);
	}

	__duel__ const vec3<Float> operator-(const vec3<Float>& v, const Float& n)
	{
		return vec3<Float>(v.x - n, v.y - n, v.z - n);
	}
	__duel__ const vec3<Float> operator-(const vec3<Float>& v0, const vec3<Int>& v1)
	{
		return vec3<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

	__duel__ const vec3<Float> operator-(const vec3<Int>& v0, const vec3<Float>& v1)
	{
		return vec3<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

	__duel__ const vec3<Float> operator-(const vec3<Float>& v0, const vec3<Float>& v1)
	{
		return vec3<Float>(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
	}

	__duel__ const vec3<Float>& operator-=(vec3<Float>& v, const Float& n)
	{
		v.x -= n;
		v.y -= n;
		v.z -= n;
		return v;
	}

	__duel__ const vec3<Float>& operator-=(vec3<Float>& v0, const vec3<Int>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		return v0;
	}
	__duel__ const vec3<Float>& operator-=(vec3<Float>& v0, const vec3<Float>& v1)
	{
		v0.x -= v1.x;
		v0.y -= v1.y;
		v0.z -= v1.z;
		return v0;
	}

#pragma endregion

#pragma region vec3f multiply operation

	__duel__ const vec3<Float> operator*(const Float& n, const vec3<Float>& v)
	{
		return vec3<Float>(n * v.x, n * v.y, n * v.z);
	}

	__duel__ const vec3<Float> operator*(const vec3<Float>& v, const Float& n)
	{
		return vec3<Float>(v.x * n, v.y * n, v.z * n);
	}
	__duel__ const vec3<Float> operator*(const vec3<Float>& v0, const vec3<Int>& v1)
	{
		return vec3<Float>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	__duel__ const vec3<Float> operator*(const vec3<Int>& v0, const vec3<Float>& v1)
	{
		return vec3<Float>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	__duel__ const vec3<Float> operator*(const vec3<Float>& v0, const vec3<Float>& v1)
	{
		return vec3<Float>(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
	}

	__duel__ const vec3<Float>& operator*=(vec3<Float>& v, const Float& n)
	{
		v.x *= n;
		v.y *= n;
		v.z *= n;
		return v;
	}

	__duel__ const vec3<Float>& operator*=(vec3<Float>& v0, const vec3<Int>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		return v0;
	}
	__duel__ const vec3<Float>& operator*=(vec3<Float>& v0, const vec3<Float>& v1)
	{
		v0.x *= v1.x;
		v0.y *= v1.y;
		v0.z *= v1.z;
		return v0;
	}

#pragma endregion

#pragma region vec3f divide operation

	__duel__ const vec3<Float> operator/(const Float& n, const vec3<Float>& v)
	{
		CHECK(v.x != 0, "(Float)n/vec3<Float>(v).x: can not devide 0!");
		CHECK(v.y != 0, "(Float)n/vec3<Float>(v).y: can not devide 0!");
		CHECK(v.z != 0, "(Float)n/vec3<Float>(v).z: can not devide 0!");

		return vec3<Float>(n / v.x, n / v.y, n / v.z);
	}

	__duel__ const vec3<Float> operator/(const vec3<Float>& v, const Float& n)
	{
		CHECK(n != 0, "vec3<Float>(v)/(Float)n: can not devide 0!");

		return vec3<Float>(v.x / n, v.y / n, v.z / n);
	}
	__duel__ const vec3<Float> operator/(const vec3<Float>& v0, const vec3<Int>& v1)
	{
		CHECK(v1.x != 0, "vec3<Float>(v0).x/vec3<Int>(v1).x: can not devide 0!");
		CHECK(v1.y != 0, "vec3<Float>(v0).y/vec3<Int>(v1).y: can not devide 0!");
		CHECK(v1.z != 0, "vec3<Float>(v0).z/vec3<Int>(v1).z: can not devide 0!");

		return vec3<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	__duel__ const vec3<Float> operator/(const vec3<Int>& v0, const vec3<Float>& v1)
	{
		CHECK(v1.x != 0, "vec3<Int>(v0).x/vec3<Float>(v1).x: can not devide 0!");
		CHECK(v1.y != 0, "vec3<Int>(v0).y/vec3<Float>(v1).y: can not devide 0!");
		CHECK(v1.z != 0, "vec3<Int>(v0).z/vec3<Float>(v1).z: can not devide 0!");

		return vec3<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	__duel__ const vec3<Float> operator/(const vec3<Float>& v0, const vec3<Float>& v1)
	{
		CHECK(v1.x != 0, "vec3<Float>(v0).x/vec3<Float>(v1).x: can not devide 0!");
		CHECK(v1.y != 0, "vec3<Float>(v0).y/vec3<Float>(v1).y: can not devide 0!");
		CHECK(v1.z != 0, "vec3<Float>(v0).z/vec3<Float>(v1).z: can not devide 0!");

		return vec3<Float>(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
	}

	__duel__ const vec3<Float>& operator/=(vec3<Float>& v, const Float& n)
	{
		CHECK(n != 0, "vec3<Float>(v)/=(Float)n: can not devide 0!");

		v.x /= n;
		v.y /= n;
		v.z /= n;
		return v;
	}

	__duel__ const vec3<Float>& operator/=(vec3<Float>& v0, const vec3<Int>& v1)
	{
		CHECK(v1.x != 0, "vec3<Float>(v0).x/=vec3<Int>(v1).x: can not devide 0!");
		CHECK(v1.y != 0, "vec3<Float>(v0).y/=vec3<Int>(v1).y: can not devide 0!");
		CHECK(v1.z != 0, "vec3<Float>(v0).z/=vec3<Int>(v1).z: can not devide 0!");

		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}
	__duel__ const vec3<Float>& operator/=(vec3<Float>& v0, const vec3<Float>& v1)
	{
		CHECK(v1.x != 0, "vec3<Float>(v0).x/=vec3<Float>(v1).x: can not devide 0!");
		CHECK(v1.y != 0, "vec3<Float>(v0).y/=vec3<Float>(v1).y: can not devide 0!");
		CHECK(v1.z != 0, "vec3<Float>(v0).z/=vec3<Float>(v1).z: can not devide 0!");

		v0.x /= v1.x;
		v0.y /= v1.y;
		v0.z /= v1.z;
		return v0;
	}

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