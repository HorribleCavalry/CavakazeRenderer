#include "../CudaSTD/CudaUtility.cuh"


namespace CUM
{
	template<typename T>
	class primitiveVector
	{
	private:
		T** ptrList;
		Uint size;
	public:
		__duel__ primitiveVector() : ptrList(new T*), size(0) {}

		__duel__ ~primitiveVector()
		{
			for (Uint i = 0; i < size; i++)
			{
				//ptrList
			}
		}
	};
}