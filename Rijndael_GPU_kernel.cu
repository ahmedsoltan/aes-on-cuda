/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This sample is a templatized version of the template project.
 * It also shows how to correctly templatize dynamically allocated shared
 * memory arrays.
 * Device code.
 */

#ifndef _RIJNDAEL_GPU_KERNEL_H_
#define _RIJNDAEL_GPU_KERNEL_H_

#include <stdio.h>

#define xmult(a) ((a)<<1) ^ (((a)&128) ? 0x01B : 0)

// const device memory
__device__ __constant__ unsigned char d_byte_sub_const[256];
__device__ __constant__ unsigned char d_gf2_8_inv_const[256];
__device__ __constant__ unsigned char d_inv_byte_sub_const[256];
__device__ __constant__ unsigned long d_Rcon_const[60];
__device__ __constant__ unsigned char d_shift_row_map_const[16];
__device__ __constant__ unsigned char d_inv_shift_row_map_const[16];

__device__ __constant__ unsigned char d_mult_by_2_const[256];
__device__ __constant__ unsigned char d_mult_by_3_const[256];
__device__ __constant__ unsigned char d_mult_by_9_const[256];
__device__ __constant__ unsigned char d_mult_by_11_const[256];
__device__ __constant__ unsigned char d_mult_by_13_const[256];
__device__ __constant__ unsigned char d_mult_by_14_const[256];

__device__ __constant__ unsigned char d_key_const[480];

struct SharedMemory
{
    __device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }  
};

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void testKernel( int* g_idata, int* g_odata) 
{
  // Shared mem size is determined by the host app at run time
  SharedMemory smem;
  int* sdata = smem.getPointer();

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  //CUPRINTF("\tValue is:%d\n", tid);
  CUPRINTF("\tbyteSub[0] = 0x%x\tbyteSub[255] = 0x%x\n",d_byte_sub_const[0],d_byte_sub_const[255]);  
  CUPRINTF("\tgf2_8_inv[0] = 0x%x\tgf2_8_inv[255] = 0x%x\n",d_gf2_8_inv_const[0],d_gf2_8_inv_const[255]);  
  CUPRINTF("\tinv_byte_sub[0] = 0x%x\tinv_byte_sub[255] = 0x%x\n",d_inv_byte_sub_const[0],d_inv_byte_sub_const[255]);  
  CUPRINTF("\tRcon[0] = 0x%x\tRcon[59] = 0x%x\n",d_Rcon_const[0],d_Rcon_const[59]);    

  // read in input data from global memory
  sdata[tid] = g_idata[tid];
  __syncthreads();

  // perform some computations
  sdata[tid] = (int) num_threads * sdata[tid];
  __syncthreads();

  // write data to global memory
  g_odata[tid] = sdata[tid];
}

__global__ void d_Round( unsigned char* g_state_idata, unsigned char* g_state_odata, unsigned char* g_key, int Nr , int RowSize) 
{
	__device__ __shared__ unsigned char s_state[64];
	//__device__ __shared__ unsigned char s_temp_state[64];
	//__device__ __shared__ char key[480]; //4*8*15

	// access thread id
	const unsigned int tid = threadIdx.x;
	// access number of threads in this block
	const unsigned int num_threads = blockDim.x;
	// thread row index
	const unsigned int Row = tid/RowSize;
	// thread col index
	const unsigned int Col = tid%RowSize;
	//const unsigned int keyIndex = Nr*RowSize*4+tid;
	const unsigned int Row1stIndex = Row * RowSize;
	//const unsigned int Col1stIndex = Col * 4;

	s_state[tid] = g_state_idata[tid];
	__syncthreads();

	// Round0:
	s_state[tid] ^= g_key[tid];
	__syncthreads();
	//CUPRINTF("Round0: state[%d] = 0x%x\n",tid,s_state[tid]);

	for (int i = 1; i < Nr; i++) {
		//CUPRINTF("Round%d: state[%d] = 0x%x\n",i,tid,s_state[tid]);
		s_state[tid] = d_byte_sub_const[s_state[tid]];
		__syncthreads();
		//CUPRINTF("after ByteSub: state[%d] = 0x%x\n",tid,s_state[tid]);  
		s_state[tid] = s_state[d_shift_row_map_const[tid]];
		__syncthreads();
		//CUPRINTF("after shiftRows: state[%d] = 0x%x\n",tid,s_state[tid]);  
		//s_temp_state[tid] = s_state[tid];
		//__syncthreads();
		s_state[tid] = s_state[tid] ^ s_state[Row1stIndex] ^ s_state[Row1stIndex+1] ^ s_state[Row1stIndex+2] ^ s_state[Row1stIndex+3] ^ xmult(s_state[tid]) ^ xmult(s_state[Row1stIndex+((tid+1)%RowSize)]);  
		__syncthreads();
		//CUPRINTF("after MixColumn: state[%d] = 0x%x\n",tid,s_state[tid]);
		s_state[tid] ^= g_key[i*RowSize*4+tid];
		__syncthreads();		
		//CUPRINTF("after AddRoundKey: state[%d] = 0x%x\n",tid,s_state[tid]);  
	}

	s_state[tid] = d_byte_sub_const[s_state[tid]];
	__syncthreads();
	s_state[tid] = s_state[d_shift_row_map_const[tid]];
	__syncthreads();
	s_state[tid] ^= g_key[Nr*RowSize*4+tid];
	__syncthreads();		

	// write data to global memory
	g_state_odata[tid] = s_state[tid];
	
}
__global__ void d_inv_Round( unsigned char* g_state_idata, unsigned char* g_state_odata, int Nr , int RowSize) 
{
	__device__ __shared__ unsigned char s_state[64];
	//__device__ __shared__ unsigned char s_temp_state[64];
	//__device__ __shared__ char key[480]; //4*8*15

	// access thread id
	const unsigned int tid = threadIdx.x;
	// access number of threads in this block
	const unsigned int num_threads = blockDim.x;
	// thread row index
	const unsigned int Row = tid/RowSize;
	// thread col index
	const unsigned int Col = tid%RowSize;
	//const unsigned int keyIndex = Nr*RowSize*4+tid;
	const unsigned int Row1stIndex = Row * RowSize;
	//const unsigned int Col1stIndex = Col * 4;

	s_state[tid] = g_state_idata[tid];
	__syncthreads();

	// AddRoundKey(Nr)
	s_state[tid] ^= d_key_const[Nr*RowSize*4+tid];
	__syncthreads();
	// InvShiftRow(Nr)
	s_state[tid] = s_state[d_inv_shift_row_map_const[tid]];
	__syncthreads();
	//InvByteSub(Nr)
	s_state[tid] = d_inv_byte_sub_const[s_state[tid]];
	__syncthreads();
	for (int i = Nr-1; i > 0; i--) {
		//AddRoundKey(i)
		s_state[tid] ^= d_key_const[i*RowSize*4+tid];
		__syncthreads();	
		//InvMixColumn(i)
		s_state[tid] = d_mult_by_14_const[s_state[tid]]  
					 ^ d_mult_by_11_const[s_state[Row1stIndex+(tid+1)%RowSize]]
					 ^ d_mult_by_13_const[s_state[Row1stIndex+(tid+2)%RowSize]]
					 ^ d_mult_by_9_const[s_state[Row1stIndex+(tid+3)%RowSize]];
		__syncthreads();
		//InvShiftByte(i)
		s_state[tid] = s_state[d_inv_shift_row_map_const[tid]];
		__syncthreads();
		//InvByteSub(i)
		s_state[tid] = d_inv_byte_sub_const[s_state[tid]];
		__syncthreads();
	}
	// AddRoundKey(0)
	s_state[tid] ^= d_key_const[tid];
	__syncthreads();

	// write data to global memory
	g_state_odata[tid] = s_state[tid];
}

__global__ void d_inv_Round_multiBlock( unsigned char* g_state_idata, unsigned char* g_state_odata, int Nr , int RowSize) 
{
	//allocate shared memory
	__device__ __shared__ unsigned char s_state[256];	

	// access number of threads in this block
	const unsigned int num_threads = blockDim.x * blockDim.y; // = 256

	// block shared memory location
	const unsigned int s_mem_idx = blockIdx.x * num_threads;

	// access thread id
	const unsigned int tid = threadIdx.x + threadIdx.y * 16;	

	// access thread row first idx
	const unsigned int state_offset = threadIdx.y * 16;	

	// aaccess thread id within cypher block
	const unsigned int ctid = tid % 16;

	// thread row index
	const unsigned int Row = ctid/RowSize;
	// thread col index
	const unsigned int Col = ctid%RowSize;
	//const unsigned int keyIndex = Nr*RowSize*4+tid;
	const unsigned int Row1stIndex = Row * RowSize;

	//CUPRINTF("kernel vars: gridDim.x = %d , gridDim.y = %d, blockDim.x = %d, blockDim.y = %d, blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = $d, threadIdx.y = %d\n",gridDim.x, gridDim.y,blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
	CUPRINTF("registers values:  num_threads = %d, s_mem_idx = %d, tid = %d, state_offset = &d, ctid = %d, Row = %d, Col = $d, Row1stIndex = %d \n",num_threads,s_mem_idx,tid,state_offset,ctid,Row,Col);

	s_state[tid] = g_state_idata[tid + s_mem_idx];
	__syncthreads();

	// AddRoundKey(Nr)
	s_state[tid] ^= d_key_const[Nr*RowSize*4+ctid];
	__syncthreads();
	// InvShiftRow(Nr)
	s_state[tid] = s_state[d_inv_shift_row_map_const[ctid] + state_offset];
	__syncthreads();
	//InvByteSub(Nr)
	s_state[tid] = d_inv_byte_sub_const[s_state[tid]];
	__syncthreads();
	for (int i = Nr-1; i > 0; i--) {
		//AddRoundKey(i)
		s_state[tid] ^= d_key_const[i*RowSize*4+ctid];
		__syncthreads();	
		//InvMixColumn(i)
		s_state[tid] = d_mult_by_14_const[s_state[tid]]  
				^ d_mult_by_11_const[s_state[Row1stIndex + (ctid+1)%RowSize + state_offset]]
				^ d_mult_by_13_const[s_state[Row1stIndex + (tid+2)%RowSize + state_offset]]
				^ d_mult_by_9_const[s_state[Row1stIndex + (tid+3)%RowSize + state_offset]];
		__syncthreads();
		//InvShiftByte(i)
		s_state[tid] = s_state[d_inv_shift_row_map_const[ctid] + state_offset];
		__syncthreads();
		//InvByteSub(i)
		s_state[tid] = d_inv_byte_sub_const[s_state[tid]];
		__syncthreads();
	}
	// AddRoundKey(0)
	s_state[tid] ^= d_key_const[ctid];
	__syncthreads();

	// write data to global memory
	g_state_odata[tid + s_mem_idx] = s_state[tid];
}
#endif // #ifndef_RIJNDAEL_GPU_KERNEL_H_
