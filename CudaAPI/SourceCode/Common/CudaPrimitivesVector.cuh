#ifndef __CUDAPRIMITIVESVECTOR__CUH__
#define __CUDAPRIMITIVESVECTOR__CUH__

#include "../CudaSTD/CudaUtility.cuh"
#include "Tools.cuh"


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
		__duel__ PrimitiveVector(const PrimitiveVector& vec) : ptrList(vec.ptrList), capacity(vec.capacity), size(vec.size) {}
		__duel__ PrimitiveVector(const T& val) : ptrList(new T*[1]), capacity(1), size(1) { ptrList[0] = &val; }
		__duel__ ~PrimitiveVector()
		{
			//if (ptrList)
			//	delete[] ptrList;
			//ptrList = nullptr;
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
		__duel__ const Int Size()
		{
			return size;
		}
	public:
		__duel__ void GetMinMax(CUM::Point3f& pMin, CUM::Point3f& pMax)
		{

		}
		__duel__ const bool HitTest(Ray& ray)
		{
			Float times = FLT_MAX;
			Ray sampleRay(ray);
			Ray resultRay(ray);
			Bool isHit = false;
			for (Int i = 0; i < size; i++)
			{
				sampleRay = ray;
				if (ptrList[i]->HitTest(sampleRay))
				{
					isHit = true;
					if (sampleRay.record.times < times)
					{
						times = sampleRay.record.times;
						resultRay = sampleRay;
					}
				}
				
			}
			ray = resultRay;
			return isHit;
		}

	public:
		__host__ PrimitiveVector* copyToDevice()
		{
			PrimitiveVector vecInsWithDevicePtr(*this);

			T** ptrListHost = new T*[capacity];
			for (Int i = 0; i < size; i++)
			{
				ptrListHost[i] = ptrList[i]->copyToDevice();
			}

			T** ptrListDevice;
			cudaMalloc(&ptrListDevice, capacity * sizeof(T*));
			cudaMemcpy(ptrListDevice, ptrListHost, size * sizeof(T*), cudaMemcpyKind::cudaMemcpyHostToDevice);
			delete[] ptrListHost;

			vecInsWithDevicePtr.ptrList = ptrListDevice;

			PrimitiveVector* vecDevice = CudaInsMemCpyHostToDevice(&vecInsWithDevicePtr);
			return vecDevice;
		}
		__duel__ void Release()
		{
			CHECK(ptrList, "PrimitiveVector::Release() error:: ptrList can not be nullptr");
			custd::OStream os;
			os << "Called PrimitiveVector::Release()!\n";
			for (Int i = 0; i < size; i++)
			{
				if (ptrList[i])
				{
					ptrList[i]->Release();
					delete ptrList[i];
					ptrList[i] = nullptr;
				}
			}
			if (ptrList)
			{
				delete[] ptrList;
				ptrList = nullptr;
			}
		}
	};
}

#endif // !__CUDAPRIMITIVESVECTOR__CUH__