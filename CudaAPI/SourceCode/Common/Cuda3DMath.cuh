#ifndef __CUDA3DMATH__CUH__
#define __CUDA3DMATH__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "../CudaSTD/cuvector.cuh"

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
	__duel__ const vec2<T>(const vec2<U>& v) : x(v.x), y(v.y) {}

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

//template<typename T, typename U>
//auto operator+(const vec2<T>& v0, const vec2<U>& v1)
//{
//	return vec2(v0.x + v1.x, v0.y + v1.y);
//}
#endif // !__CUDA3DMATH__CUH__
