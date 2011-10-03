void Rijndael_GPU::EncryptBlock(const unsigned char * datain1, unsigned char * dataout1, const unsigned char * states)
{ 
	const unsigned long * datain = reinterpret_cast<const unsigned long*>(datain1);
	unsigned long * dataout = reinterpret_cast<unsigned long*>(dataout1);

	//TODO: move these to initGPU
	int devID;
	cudaDeviceProp deviceProps;
	devID = cutGetMaxGflopsDeviceId();
	cudaSetDevice( devID );

	// get number of SMs on this GPU
	cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
	//printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

	unsigned int num_threads = 16;	

	// allocate device memory
	size_t memsize = state_size;

	unsigned char* d_idata;
	cutilSafeCall( cudaMalloc( (void**) &d_idata, memsize));
	// copy host memory to device
	cutilSafeCall( cudaMemcpy( d_idata, datain1, memsize, cudaMemcpyHostToDevice));

	unsigned char* d_ext_key;
	int key_size = Nb*(Nr+1)*4;
	cutilSafeCall( cudaMalloc( (void**) &d_ext_key, key_size));
	// copy host memory to device
	cutilSafeCall( cudaMemcpy( d_ext_key, W, key_size, cudaMemcpyHostToDevice));

	// allocate device memory for result
	unsigned char* d_odata;
	cutilSafeCall( cudaMalloc( (void**) &d_odata, memsize));

	// setup execution parameters
	dim3  grid( 1, 1, 1);
	dim3  threads( num_threads, 1, 1);

	unsigned int timer = 0;
	cutilCheckError( cutCreateTimer( &timer));
	cutilCheckError( cutStartTimer( timer));

	//////////////////////////////////////////////////////////////////
	//Architectures with compute capability 1.x, function
	//cuPrintf() is used. Otherwise, function printf() is called.
	bool use_cuPrintf = (deviceProps.major < 2);

	if (use_cuPrintf)
	{
		//Initializaton, allocate buffers on both host
		//and device for data to be printed.
		cudaPrintfInit();
	}
	/////////////////////////////////////////////////////////////////

	//printf("launching GPU kernel...\n");

	// execute the kernel
	d_Round<<< grid, threads >>>( d_idata, d_odata, d_ext_key, Nr, Nb);
	cutilDeviceSynchronize(); // ???

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");


	/////////////////////////////////////////////////////////////
	if (use_cuPrintf)
	{
		//Dump current contents of output buffer to standard 
		//output, and origin (block id and thread id) of each line 
		//of output is enabled(true).
		cudaPrintfDisplay(stdout, true);

		//Free allocated buffers by cudaPrintfInit().
		cudaPrintfEnd();
	}
	//////////////////////////////////////////////////////////////



	cutilSafeCall( cudaMemcpy( state, d_odata, state_size,	cudaMemcpyDeviceToHost) );
	cutilCheckError( cutStopTimer( timer));
	//printf( "GPU Processing time: %f (ms)\n", cutGetTimerValue( timer));
	cutilCheckError( cutDeleteTimer( timer));

	memcpy(dataout,state,state_size);

	cutilSafeCall(cudaFree(d_idata));
	cutilSafeCall(cudaFree(d_ext_key));
	cutilSafeCall(cudaFree(d_odata));

	cutilDeviceReset();
} // Encrypt