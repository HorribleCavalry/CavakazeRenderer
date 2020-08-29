#include "CudaAPI.cuh"
#include <array>
#include <vector>


//To solve the problem that can not use "CHECK" from another file in __global__ function, just choose the project setting->CUDA C/C++->Generate Relocatable Device Code.
//Refercenced website: https://www.cnblogs.com/qpswwww/p/11646593.html

__duel__ void testDuel()
{
}

__global__ void kernel()
{
	vec2i vi;
	vec2f vf;

	vf.x = 1.5;
	vf.y = 1.5;
	printf("%f\n", vf.x);
	vi = vf;
	printf("%d\n", vi[0]);

	vi[0] = 2;
	printf("%d\n", vi[0]);

}


int main()
{
	kernel << <1, 1 >> > ();
}