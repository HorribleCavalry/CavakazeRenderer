#include "cuvector.cuh"

namespace custd
{
	template<typename T>
	cuvector<T>::cuvector()
		:data(new T),Size(1), BackIdx(0), Capacity(1)
	{
	}
	template<typename T>
	cuvector<T>::~cuvector()
	{
		if (data)
			delete[] data;
	}
	template<typename T>
	__host__ __device__ Uint cuvector<T>::size()
	{
		return Size;
	}
	template<typename T>
	__host__ __device__ Uint cuvector<T>::capacity()
	{
		return Capacity;
	}
	template<typename T>
	void cuvector<T>::push_back(const T & val)
	{
		CHECK(data, "The data in current cuvector is nullptr!");
		Int debugArray[3];
		debugArray[0] = Size;
		debugArray[1] = BackIdx;
		debugArray[2] = Capacity;
		CHECK(Capacity > 0, "The capacity in current cuvector is a negative value!", debugArray, 3);
		CHECK(Capacity >= Size, "The capacity must be greater equal to Size in current cuvector!", debugArray, 3);
		CHECK(Size == BackIdx + 1, "The BackIdx is not match Size in current cuvector!", debugArray, 3);

		if (Size<=Capacity)
		{
			data[BackIdx + 1] = val;
		}
		if(Size==Capacity)
		{
			Capacity = Capacity <= 0 ? 1 : 2 * Capacity;
			T* newData = new T[Capacity];

			for (Uint i = 0; i < Size; i++)
			{
				newData[i] = (*this)[i];
			}
			newData[Size] = val;

			if (data) delete[] data;
			data = newData;
		}
		++Size;
		++BackIdx;

	}
	template<typename T>
	T & cuvector<T>::operator[](Int idx)
	{
		CHECK(idx < Size, "The input index in current cuvector is out of range!");
		CHECK(idx > 0, "The input index in a cuvector can not be a negative value!");
		return data[idx]
	}
}