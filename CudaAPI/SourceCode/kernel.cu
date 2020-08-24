#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void kernel()
{
	auto i = 11;
	auto op = i + 12.0f;
	//thrust::device_vector<int> vec;
	//vec.push_back(op);
	//int temp = vec[0];
	//printf("%d", temp);
	//thrust library can not use in kernel.
	//Referenced: https://forums.developer.nvidia.com/t/using-std-vector-in-cuda-kernel-its-posible-to-use-a-std-vector-inside-cuda-kernel/16307/3
	//Referenced: https://docs.nvidia.com/cuda/thrust/index.html
}

int main()
{
	thrust::host_vector<int> H(4);

	H[0] = 14;
	H[1] = 20;
	H[2] = 38;
	H[3] = 46;

	std::cout << "H has size: " << H.size() << std::endl;

	for each (auto var in H)
	{
		std::cout << var << std::endl;
	}
	thrust::device_vector<int> device_vec;
	device_vec = H;
	kernel << <1, 1>> > ();
}