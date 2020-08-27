#include "cuarray.cuh"

namespace custd
{
	template<typename T, int size>
	cuarray<T, size>::cuarray()
	{
		data = new T[size];
	}
}