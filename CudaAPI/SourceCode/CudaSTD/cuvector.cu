﻿#include "cuvector.cuh"

namespace custd
{

	template<typename T>
	__duel__ Uint cuvector<T>::size()
	{
		return Size;
	}
	template<typename T>
	__duel__ Uint cuvector<T>::capacity()
	{
		return Capacity;
	}

}