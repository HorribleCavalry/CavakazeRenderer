#ifndef __CUVACTOR__CUH__
#define __CUVACTOR__CUH__

#include "CudaUtility.cuh"
#include "cuiostream.cuh"

namespace custd
{
	template<typename T>
	class cuvector
	{
	private:
		T* data;
		Uint Size;
		Uint Capacity;
	public:
		__duel__ cuvector()
			:data(new T), Size(0), Capacity(0)
		{
			CHECK(data, "The data initialization in construct function failed");
		}
		__duel__ cuvector(const Uint& _Size)
			:data(new T[_Size]), Size(_Size), Capacity(_Size)
		{
			CHECK(data, "The data initialization in construct function failed");
		}

		__duel__ void Release()
		{
			if (data)
			{
				if (Size == 0)
					delete data;
				else
					delete[] data;
				data = nullptr;
			}

			data = new T;
			Size = 0;
			Capacity = 0;
		}

		__duel__ ~cuvector()
		{
			if (data)
			{
				if (Size == 0)
					delete data;
				else
					delete[] data;
				data = nullptr;
			}
			Capacity = 0;
			Size = 0;
		}

		__duel__ Int size() const
		{
			return Size;
		}
		__duel__ Int capacity() const
		{
			return Capacity;
		}

		__duel__ void push_back(const T& val)
		{
			if (Size == Capacity)
			{
				T* newData = nullptr;
				Capacity = Capacity <= 0 ? 1 : 2 * Capacity;
				newData = new T[Capacity];

				for (Uint i = 0; i < Size; i++)
				{
					newData[i] = (*this)[i];
				}

				if (data)
				{
					if (Size == 0)
						delete data;
					else
						delete[] data;
				}
				data = newData;
				newData = nullptr;
			}
			data[Size] = val;
			++Size;
		}
		

		__duel__ T& operator[](Int idx) const
		{
			CHECK(idx < Size, "The input index in current cuvector is out of range!");
			CHECK(idx >= 0, "The input index in a cuvector can not be a negative value!");
			return data[idx];
		}

	};
}

#endif // !__CUVACTOR__CUH__
