#ifndef __CUVACTOR__CUH__
#define __CUVACTOR__CUH__

#include "CudaUtility.cuh"


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
		__host__ __device__ cuvector()
			:data(new T), Size(0), Capacity(0)
		{
			CHECK(data, "The data initialization in construct function failed");
		}
		__host__ __device__ cuvector(const Uint& _Size)
			:data(new T[_Size]), Size(_Size), Capacity(_Size)
		{
			CHECK(data, "The data initialization in construct function failed");

		}
		__host__ __device__ ~cuvector()
		{
			if (data)
				delete[] data;
		}

		__host__ __device__ Uint size();
		__host__ __device__ Uint capacity();

		__host__ __device__ void push_back(const T& val)
		{
			CHECK(data, "The data in current cuvector is nullptr!");
			//Int* debugArray = new Int[3];
			//debugArray[0] = Size;
			//debugArray[1] = BackIdx;
			//debugArray[2] = Capacity;
			//check(Capacity > 0, "The capacity in current cuvector can not be a negative value!", debugArray, 3);
			//check(Capacity >= Size, "The capacity must be greater equal to Size in current cuvector!", debugArray, 3);
			//check(Size == BackIdx + 1, "The BackIdx is not match Size in current cuvector!", debugArray, 3);

			//CHECK(Capacity > 0, "The capacity in current cuvector can not be a negative value!", debugArray, 3);
			//CHECK(Capacity >= Size, "The capacity must be greater equal to Size in current cuvector!", debugArray, 3);
			//CHECK(Size == BackIdx + 1, "The BackIdx is not match Size in current cuvector!", debugArray, 3);

			T* newData = nullptr;

			if (Size == Capacity)
			{
				Capacity = Capacity <= 0 ? 1 : 2 * Capacity;
				newData = new T[Capacity];

				for (Uint i = 0; i < Size; i++)
				{
					newData[i] = (*this)[i];
				}

				if (data) delete[] data;
				data = newData;
				newData = nullptr;
			}
			data[Size] = val;
			++Size;
		}
		

		__host__ __device__ T& operator[](Int idx)
		{
			CHECK(idx < Size, "The input index in current cuvector is out of range!");
			CHECK(idx >= 0, "The input index in a cuvector can not be a negative value!");
			return data[idx];
		}

	};
}

#endif // !__CUVACTOR__CUH__
