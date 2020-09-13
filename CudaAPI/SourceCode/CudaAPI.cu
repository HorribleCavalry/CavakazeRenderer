#include "Common/Cuda3DMath.cuh"
#include "Common/CudaPrimitivesVector.cuh"

//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html



class Person
{
public:
	__duel__ virtual void callType()
	{
		custd::OStream os;
		os << "I'm a person!" << custd::endl;
	}
};

class Student : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os<<"I'm a student!" << custd::endl;
	}
};

class Teacher : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a teacher!" << custd::endl;
	}
};

class Farmer : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a Farmer!" << custd::endl;
	}
};

class Heacker : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a Heacker!" << custd::endl;
	}
};

class Worker : public Person
{
public:
	__duel__ virtual void callType() override
	{
		custd::OStream os;
		os << "I'm a Worker!" << custd::endl;
	}
};

__global__ void kernel()
{
	Person per0;
	Student stu;
	Farmer far;
	Heacker hea;
	Worker wor;

	CUM::PrimitiveVector<Person> list;
	list.push_back(per0);
	list.push_back(stu);
	list.push_back(far);
	list.push_back(hea);
	list.push_back(wor);
	for (Int i = 0; i < 5; i++)
	{
		list[i].callType();
	}

	CUM::vec4f vf4(2.0f);
	auto test = CUM::normalize(vf4);
	CUM::vec3f vf41(2.0f);
	auto test1 = CUM::normalize(vf41);
}

//__duel__ CUM::vec4<Float>&& reR()
//{
//	return CUM::vec4<Float>();
//}

int main()
{
	//Person** prList = new Person*[5];

	//Person per0;
	//Student stu;
	//Farmer far;
	//Heacker hea;
	//Worker wor;
	//prList[0] = &per0;
	//prList[1] = &stu;
	//prList[2] = &far;
	//prList[3] = &hea;
	//prList[4] = &wor;
	//
	//for (int i = 0; i < 5; i++)
	//{
	//	prList[i]->callType();
	//}

	//CUM::PrimitiveVector<Person> list;
	//list.push_back(per0);
	//list.push_back(stu);
	//list.push_back(far);
	//list.push_back(hea);
	//list.push_back(wor);
	//for (Int i = 0; i < 5; i++)
	//{
	//	list[i].callType();
	//}

	kernel << <1, 1 >> > ();

	CUM::vec2i vi0;
	CUM::vec2i vi1(1.0f, 2.0f);
	CUM::vec2f vf0;
	CUM::vec2f vf1(1.0f, 2.0f);
	CUM::vec4f vf4(2.0f);
	auto test =CUM::normalize(vf4);
	CUM::vec3f vf41(2.0f);
	auto test1 = CUM::normalize(vf41);

	Person per0;
	Student stu;
	Farmer far;
	Heacker hea;
	Worker wor;

	CUM::PrimitiveVector<Person> list;
	list.push_back(per0);
	list.push_back(stu);
	list.push_back(far);
	list.push_back(hea);
	list.push_back(wor);
	for (Int i = 0; i < 5; i++)
	{
		list[i].callType();
	}

	//Int ni = 4;
	//Float nf = 4.0;
	//auto tempvi0 = vi0 + ni;
	//auto tempvi1 = ni + vi0;
	//auto tempvi2 = vi0 + nf;
	//auto tempvi3 = nf + vi0;
	//auto tempvi4 = vi0 + vi1;
	//auto tempvi5 = vi0 + vf1;
	//auto mat = CUM::Mat4x4_identity;
	//CUM::Mat4x4i mati(5);
	//mati += mat;
	//mati -= mat;
	//mati / mat;
	//CUM::Mat4x4f matf;
	//matf /= mat;
}