#include "../CudaSTD/CudaUtility.cuh"


namespace CUM
{
	template<typename T>
	class PrimitiveVector
	{
	private:
		T** ptrList = nullptr;
		Int capacity = 0;
		Int size = 0;
	public:
		__duel__ PrimitiveVector() : ptrList(nullptr), capacity(0), size(0) {}
		__duel__ PrimitiveVector(const T& val) : ptrList(new T*[1]), capacity(1), size(1) { ptrList[0] = &val; }
		__duel__ ~PrimitiveVector()
		{
			if (ptrList)
				delete[] ptrList;
			ptrList = nullptr;
		}
	public:
		__duel__ void push_back(T& val)
		{
			++size;
			CHECK(size <= capacity + 1, "PrimitiveVector:endIdx can not greater than size!");

			if (size == capacity + 1)
			{
				Int newCapacity = capacity <= 0 ? 1 : 2 * capacity;
				T** newList = new T*[newCapacity];

				for (Int i = 0; i < capacity; i++)
				{
					newList[i] = ptrList[i];
				}
				if (ptrList)
					delete[] ptrList;
				ptrList = newList;
				capacity = newCapacity;
			}
			ptrList[size - 1] = &val;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx < size, "PrimitiveVector::operator[](Int idx) error: idx is out of range!");
			return *(ptrList[idx]);
		}

	public:

	};
}