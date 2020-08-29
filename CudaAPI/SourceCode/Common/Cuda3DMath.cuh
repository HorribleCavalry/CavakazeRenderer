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
	__duel__ vec2() :x(0), y(0) {}

	__duel__ vec2(const T& _x, const T& _y) : x(_x), y(_y) {}


	template<typename U>
	__duel__ vec2(const vec2<U>& vec) = delete;

	template<typename U>
	__duel__ vec2(vec2<U>&& vec) = delete;

	template<typename U>
	__duel__ vec2<T>& operator=(const vec2<U>& vec)
	{
		x = static_cast<T>(vec.x);
		y = static_cast<T>(vec.y);
		return *this;
	}

	template<typename U>
	__duel__ vec2<T>& operator=(vec2<U>&& vec)
	{
		x = static_cast<T>(vec.x);
		y = static_cast<T>(vec.y);
		return *this;
	}

	__duel__ ~vec2() {}

public:
	__duel__ T& operator[](const Int& idx)
	{
		CHECK(idx >= 0 && idx <= 1, "The <idx> invec2<T>::operator[idx] is illegal!");
		return idx == 0 ? x : y;
	}
};

typedef vec2<Int> vec2i;
typedef vec2<Float> vec2f;

//template<typename T, typename U>
//auto operator+(const vec2<T>& v0, const vec2<U>& v1)
//{
//	return vec2(v0.x + v1.x, v0.y + v1.y);
//}
#endif // !__CUDA3DMATH__CUH__
