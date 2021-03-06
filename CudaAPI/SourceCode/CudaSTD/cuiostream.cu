﻿#include "cuiostream.cuh"
namespace custd
{
	template<typename T>
	__duel__ void print(const T & para)
	{
		//switch (typeid(para))
		//{
		//	case typeid(short) : printf("%d", para); printf("\nCalled printf(short)\n"); break;
		//	case typeid(unsigned short) : printf("%d", para); printf("\nCalled printf(unsigned short)\n"); break;

		//	case typeid(int) : printf("%d", para); printf("\nCalled printf(int)\n"); break;
		//	case typeid(unsigned int) : printf("%d", para); printf("\nCalled printf(unsigned int)\n"); break;

		//	case typeid(float) : printf("%d", para); printf("\nCalled printf(float)\n"); break;

		//	case typeid(double) : printf("%d", para); printf("\nCalled printf(double)\n"); break;
		//default: printf("\nUnknown type.\n"); break;
		//}
		printf("Nothing");
	}

	OStream::OStream()
	{
	}

	OStream::~OStream()
	{
	}
	const OStream & OStream::operator<<(const short & val) const
	{
		printf("%d", val);
		return *this;
	}
	const OStream & OStream::operator<<(const unsigned short & val) const
	{
		printf("%u", val);
		return *this;
	}
	const OStream & OStream::operator<<(const int & val) const
	{
		printf("%i", val);
		return *this;
	}
	const OStream & OStream::operator<<(const unsigned int & val) const
	{
		printf("%u", val);
		return *this;
	}
	const OStream & OStream::operator<<(const long & val) const
	{
		printf("%ld", val);
		return *this;
	}
	const OStream & OStream::operator<<(const unsigned long & val) const
	{
		printf("%lu", val);
		return *this;
	}
	const OStream & OStream::operator<<(const long long & val) const
	{
		printf("%lld", val);
		return *this;
	}
	const OStream & OStream::operator<<(const unsigned long long & val) const
	{
		printf("%llu", val);
		return *this;
	}
	const OStream & OStream::operator<<(const float & val) const
	{
		printf("%f", val);
		return *this;
	}
	const OStream & OStream::operator<<(const double & val) const
	{
		printf("%f", val);
		return *this;
	}
	const OStream & OStream::operator<<(const char & val) const
	{
		printf("%c", val);
		return *this;
	}
	const OStream & OStream::operator<<(const char * val) const
	{
		printf("%s", val);
		return *this;
	}
	const OStream & OStream::operator<<(void(*edl)()) const
	{
		(*edl)();
		return *this;
	}

	__duel__ const OStream & OStream::operator<<(const void* ptr) const
	{
		printf("%llx", (unsigned long long)ptr);
		return *this;
	}

	__device__ void kout(const int& val)
	{
		printf("%d", val);
	}

	void endl()
	{
		printf("\n");
	}


}