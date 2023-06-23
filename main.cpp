
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_DM_NEURON.cuh"
#include "CUDA_DM_RUN.cuh"
#include <cstdio>

#include <functional>
#include <iostream>
#include <vector>
#include <random>

int main()
{

	std::cout << sizeof(bool) << std::endl;
	s_neuronal_netowrk _neuronal_network;

	int nn = 1000;

	_neuronal_network.set_neuron_number(nn);

	for (int i = 0; i < nn*0.2; i++)
	{
		_neuronal_network.set_inhibtion_neuron(i);
	}

	double g = 1.3;

	std::mt19937 gen(time(NULL));
	std::uniform_real_distribution<> uni_dist(0.0, 1.0);
	std::cout << "Connection setting....." << std::endl;
	for (int i = 0; i < nn; i++)
	{
		for (int j = i + 1; j < nn; j++)
		{
			if (uni_dist(gen) < 0.01)
			{
				if (uni_dist(gen) < 0.5)
				{
					_neuronal_network.set_connection(i, j, g);
				}
				else
				{
					_neuronal_network.set_connection(j, i, g);
				}
			}
		}
	}

	double s_time = 15;
	double _time = 20 * 1000 * s_time;
	double noise_intensity = 11.5;

	_neuronal_network.set_run_param(0.05, 0, _time, noise_intensity, true, false, false);
	std::cout << "CUDA memory setting...." << std::endl;
	
	_neuronal_network.create_cuda_memory();

	std::cout << "run...." << std::endl;


	cudaEvent_t cstart, cstop;

	cudaEventCreate(&cstart);
	cudaEventCreate(&cstop);
	cudaEventRecord(cstart);

	//main function 
	_neuronal_network.cuda_run_stdp();

	cudaEventRecord(cstop);
	cudaEventSynchronize(cstop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cstart, cstop);
	std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
	cudaEventDestroy(cstart);
	cudaEventDestroy(cstop);


	_neuronal_network.save_spike_data("ptest.txt");
	


	return 1;
}