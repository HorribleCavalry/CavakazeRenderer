#include "CudaAPI.cuh"
#include <array>
#include <vector>


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

class Base
{
public:

	__host__ __device__ virtual void print()
	{
		printf("Called Base!\n");
	}
};

class Child0 : public Base
{
public:
	__host__ __device__ virtual void print() override
	{
		printf("Called Child0!\n");
	}
};

class Child1 : public Base
{
public:
	__host__ __device__ virtual void print() override
	{
		printf("Called Child1!\n");
	}
};

__global__ void kernel()
{
	//CHECK(false, "Nothing.");
	//custd::kout(23);
	//printf("Yes");
	//TestNameSpace::ost ot;
	//custd::Stream cout;
	//cout << "Yes";
	//cuda::std::
	//custd::print(3);
	//custd::Stream st;
	//st << "Yes\n";
	Base base;
	Base tem[3];
	Child0 c0;
	Child1 c1;

	Base** basePtrList = new Base*[3];
	basePtrList[0] = &base;
	basePtrList[1] = &c0;
	basePtrList[2] = &c1;

	custd::cuvector<Base> baseVec;
	baseVec.push_back(base);
	baseVec.push_back(c0);
	baseVec.push_back(c1);

	tem[0] = base;
	tem[1] = c0;
	tem[2] = c1;

	for (int i = 0; i < 3; i++)
	{
		tem[i].print();
		basePtrList[i]->print();
	}

	for (Int i = 0; i < 3; i++)
	{
		baseVec[i].print();
	}

	custd::cuvector<Uint> vec;
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
}

int main()
{

	//short shortNum = std::numeric_limits<short>::max();
	//unsigned short unsignedShortNum = std::numeric_limits<unsigned short>::max();

	//int intNum = std::numeric_limits<int>::max();
	//unsigned int unsignedIntNum = std::numeric_limits<unsigned int>::max();

	//long longNum = std::numeric_limits<long>::max();
	//unsigned long unsignedLongNum = std::numeric_limits<unsigned long>::max();

	//long long longLongNum = std::numeric_limits<long long>::max();
	//unsigned long long unsignedLongLongNum = std::numeric_limits<unsigned long long>::max();

	//float floatNum = std::numeric_limits<float>::max();
	//double doubleNum = std::numeric_limits<double>::max();

	//custd::cout << "shortNum: " << shortNum <<custd::endl;
	//custd::cout << "unsignedShortNum: " << unsignedShortNum <<custd::endl;

	//custd::cout << "intNum: " << intNum <<custd::endl;
	//custd::cout << "unsignedIntNum: " << unsignedIntNum <<custd::endl;

	//custd::cout << "longNum: " << longNum <<custd::endl;
	//custd::cout << "unsignedLongNum: " << unsignedLongNum <<custd::endl;

	//custd::cout << "longLongNum: " << longLongNum <<custd::endl;
	//custd::cout << "unsignedLongLongNum: " << unsignedLongLongNum <<custd::endl;

	//custd::cout << "floatNum: " << floatNum << custd::endl;
	//custd::cout << "doubleNum: " << doubleNum << custd::endl;

	//custd::cout << "Now the following code is called in kernel:" << custd::endl;
	//custd::initCout
	//std::array<int, 32> vals;
	//vals.
	
	custd::cuvector<Uint> vec;
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	vec.push_back(12);
	kernel <<<1, 1 >>> ();
}