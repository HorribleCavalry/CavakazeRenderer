#include "CudaUtility.cuh"

//Referenced: https://stackoverflow.com/questions/50755717/triggering-a-runtime-error-within-a-cuda-kernel

__global__ void kernel()
{
	//If paramater is true, nothing would happen.
	assert(1);
	
	//This function can not use. I don't know why.
	//asm("trap\n");
	CHECK(false, "Called this check.");
	printf("Yes\n");
}

int main()
{

	kernel << <1, 1>> > ();
	//CUDA_CHECK(cudaGetLastError());
	//CUDA_CHECK(cudaDeviceSynchronize());

}