#include "../CudaSTD/CudaUtility.cuh"


namespace CUM
{
	template<typename T>
	class PrimitiveVector
	{
	private:
		T** ptrList = nullptr;
		Int size = 0;
		Int endIdx = -1;
	public:
		__duel__ PrimitiveVector() : ptrList(nullptr), size(0), endIdx(-1) {}
		__duel__ PrimitiveVector(const T& val) : ptrList(new T*), size(1), endIdx(0) { ptrList[0] = &val; }
		__duel__ ~PrimitiveVector()
		{
			if (ptrList)
				delete[] ptrList;
			ptrList = nullptr;
		}
	public:
		__duel__ void push_back(T& val)
		{
			++endIdx;
			CHECK(endIdx <= size, "PrimitiveVector:endIdx can not greater than size!");

			if (endIdx == size)
			{
				Int newSize = size <= 0 ? 1 : 2 * size;
				T** newList = new T*[newSize];

				for (Int i = 0; i < size; i++)
				{
					newList[i] = ptrList[i];
				}
				if (ptrList)
					delete[] ptrList;
				ptrList = newList;
				size = newSize;
			}
			ptrList[endIdx] = &val;
		}

	public:
		__duel__ T& operator[](const Int& idx)
		{
			CHECK(idx >= 0 && idx < size, "PrimitiveVector::operator[](Int idx) error: idx is out of range!");
			return *(ptrList[idx]);
		}
	};
}