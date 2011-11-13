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
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <assert.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

// includes for cuda auto-complete suport
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, kernels
#include "simpleTemplates_kernel.cu"

// include timer function
#include "time_ms_64.hpp"

/////////////////////////////////////////////////////////////////////////////////
// Includes / Declerations for AES GPU code
/////////////////////////////////////////////////////////////////////////////////
#include "Rijndael_GPU.cuh"
#define _GPU
//#define _SLOW
/////////////////////////////////////////////////////////////////////////////////
// Includes / Declerations for AES host code
/////////////////////////////////////////////////////////////////////////////////
#include "Rijndael.h"
#include "AES.h"

using namespace std;

//#define __TimingTests 

// define this to test old direct slow method, else remove for fast method
//#define _SLOW
#define RANDOM_TEST_COUNT 50 // how many random tests to do
////////////////////////////////////////////////////////////////////////////////

#ifdef _SLOW
#define CRYPT Rijndael
#else
	#ifdef _GPU
	#define CRYPT Rijndael_GPU
	#endif
#endif

typedef struct
	{
	char * key;
	char * plaintext;
	char * ciphertext;
	char * e_vectors[9];
	char * d_vectors[9];
	} test_t;

// todo - add all checks from the NIST document in the header comment

test_t vectors[] = {
		// a test vector from NIST
	{   "000102030405060708090A0B0C0D0E0F", // key
	    "000102030405060708090A0B0C0D0E0F", // plaintext
		"0A940BB5416EF045F1C39458C653EA5A", // ciphertext
			{"B5C9179EB1CC1199B9C51B92B5C8159D", // encryption vectors
			"2B65F6374C427C5B2FE3A9256896755B",
			"D1015FCBB4EF65679688462076B9D6AD",
			"8E17064A2A35A183729FE59FF3A591F1",
			"D7557DD55999DB3259E2183D558DCDD2",
			"73A96A5D7799A5F3111D2B63684B1F7F",
			"1B6B853069EEFC749AFEFD7B57A04CD1",
			"107EEADFB6F77933B5457A6F08F046B2",
			"8EC166481A677AA96A14FF6ECE88C010"}, 

			{"8EC166481A677AA96A14FF6ECE88C010", // decryption vectors
			"107EEADFB6F77933B5457A6F08F046B2",
			"1B6B853069EEFC749AFEFD7B57A04CD1",
			"73A96A5D7799A5F3111D2B63684B1F7F",
			"D7557DD55999DB3259E2183D558DCDD2",
			"8E17064A2A35A183729FE59FF3A591F1",
			"D1015FCBB4EF65679688462076B9D6AD",
			"2B65F6374C427C5B2FE3A9256896755B",
			"B5C9179EB1CC1199B9C51B92B5C8159D"}},

	// a bunch of different test values
	{"00010203050607080A0B0C0D0F101112",
	"506812A45F08C889B97F5980038B8359",
	"D8F532538289EF7D06B506A4FD5BE9C9",	{0},{0}},

	{"00010203050607080A0B0C0D0F10111214151617191A1B1C",
	"2D33EEF2C0430A8A9EBF45E809C40BB6",
	"DFF4945E0336DF4C1C56BC700EFF837F", {0},{0}},

	{"50515253555657585A5B5C5D5F60616264656667696A6B6C6E6F707173747576",
	"050407067477767956575051221D1C1F",
	"7444527095838FE080FC2BCDD30847EB", {0},{0}},

	
	{"000000000000000000000000000000000200000000000000",
	"00000000000000000000000000000000",
	"5D989E122B78C758921EDBEEB827F0C0",{0},{0}},
	};

/////////////////////////////////////////////////////////////////////////////////
int g_TotalFailures = 0;

// declaration, forward
template <class T>
void runTest( int argc, char** argv, int len);

template<class T>
void
computeGold( T* reference, T* idata, const unsigned int len) 
{
    const T T_len = static_cast<T>( len);
    for( unsigned int i = 0; i < len; ++i) 
    {
        reference[i] = idata[i] * T_len;
    }
}

////////////////////////////////////////////////////////////////////////////////
// aes main methods
////////////////////////////////////////////////////////////////////////////////
void TextToHex(const char * in, char * data)
	{
	// given a text string, convert to hex data
	int val;
	while (*in)
		{
		val = *in++;
		if (val > '9')
			val = toupper(val) - 'A' + 10;
		else
			val = val - '0';
		*data = val*16;
		val = *in++;
		if (val > '9')
			val = toupper(val) - 'A' + 10;
		else
			val = val - '0';
		*data++ += val;
		}
	} // TextToHex

// test a given test vector, see that internals are working
// return false iff fails
bool TestVector(const test_t & vector, bool use_states)
	{
	bool retval = true; // assume passes
	// data sizes in bytes
	int keylen = strlen(vector.key)/2, blocklen = strlen(vector.plaintext)/2;

	CRYPT crypt;
	crypt.SetParameters(keylen*8,blocklen*8);
	unsigned char key[32], plaintext[32],ciphertext[32],temptext[32];
	unsigned char states[4096*20];



	TextToHex(vector.key,reinterpret_cast<char*>(key));
	TextToHex(vector.ciphertext,reinterpret_cast<char*>(ciphertext));
	TextToHex(vector.plaintext,reinterpret_cast<char*>(plaintext));

	if (use_states == true)
		for (int pos = 0; pos < 9; pos++)
			TextToHex(vector.e_vectors[pos],reinterpret_cast<char*>(states)+pos*16);

	crypt.StartEncryption(key);
#ifdef _SLOW
	if (use_states == true)
		crypt.EncryptBlock(plaintext,temptext,states);
	else
#endif
		crypt.EncryptBlock(plaintext,temptext);

	// check that temp = cipher
	if (memcmp(ciphertext,temptext,blocklen) != 0)
		{
		cout << "Error: encryption error\n";
		retval = false;
		}
	else
		cout << "Encryption passed\n";

	crypt.StartDecryption(key);
#ifdef _SLOW
	if (use_states == true)
		crypt.DecryptBlock(ciphertext,temptext,states);
	else
#endif
		crypt.DecryptBlock(ciphertext,temptext);

	if (memcmp(plaintext,temptext,blocklen) != 0)
		{
		cout << "Error: decryption error\n";
		retval = false;
		}
	else
		cout << "Decryption passed\n";

	return retval;
	} // TestVector

// return false iff 2 byte end values not preserved
bool CheckBuffer(const unsigned char * buf, int length)
	{
	return (0xBE == buf[0]) && (0xEF == buf[1]) && 
		   (0xBE == buf[length+2]) && (0xEF == buf[length+3]);
	} // CheckBuffer

// return false iff fails
bool RandomTest(int pos)
	{
	// data sizes in bytes
	int keylen, blocklen,datalen,mode;
	keylen   = (rand()%3)*8 + 16;
	//blocklen = (rand()%3)*8 + 16;
	blocklen = 16;

	//mode = rand()%2; // various chaining modes
	mode = 0; // ECB only!!

	assert((16 == keylen) || (24 == keylen) || (32 == keylen));
	assert((16 == blocklen) || (24 == blocklen) || (32 == blocklen));

#define MAXDATA 4096 // max length of random data
	CRYPT crypt;
	crypt.SetParameters(keylen*8,blocklen*8);
	datalen = rand()%MAXDATA;
	unsigned char key[32], plaintext[MAXDATA+40],ciphertext[MAXDATA+40],temptext[MAXDATA+40];

	cout << "Test: " << pos+1 << "  (keysize,blocksize,datalength): (" << keylen << ',' << blocklen << "," << datalen << ")\n";

	for (pos = 0; pos < keylen; pos++)
		key[pos] = rand();
	// add buffer bytes to each end to catch errors
	plaintext[0]  = 0xBE; plaintext[1]  = 0xEF;
	ciphertext[0] = 0xBE; ciphertext[1] = 0xEF;
	temptext[0]   = 0xBE; temptext[1]   = 0xEF;
	for (pos = 0; pos < datalen; pos++)
		plaintext[pos+2] = rand();
	// pad
	int padlen = blocklen - (datalen%blocklen);
	for (pos = 0; pos < padlen; pos++)
		plaintext[pos+2+datalen] = 0;
	// add buffer bytes to each end to catch errors
	pos = padlen+2+datalen;
	plaintext[pos]  = 0xBE; plaintext[pos+1]  = 0xEF;
	ciphertext[pos] = 0xBE; ciphertext[pos+1] = 0xEF;
	temptext[pos]   = 0xBE; temptext[pos+1]   = 0xEF;

#undef MAXDATA 

	int blocks = (datalen + blocklen-1)/blocklen;
	crypt.StartEncryption(key);
	crypt.Encrypt(plaintext+2,ciphertext+2,blocks,static_cast<CRYPT::BlockMode>(mode));

	crypt.StartDecryption(key);
	crypt.Decrypt(ciphertext+2,temptext+2,blocks,static_cast<CRYPT::BlockMode>(mode));

	if (memcmp(plaintext+2,temptext+2,datalen) != 0)
		{
		cout << "Error: decryption error\n";
		return false;
		}
	else if ((false == CheckBuffer(plaintext,datalen+padlen)) ||
		     (false == CheckBuffer(temptext,datalen+padlen))  ||
			 (false == CheckBuffer(ciphertext,datalen+padlen)))
		{
		cout << "Error: buffer overflow\n";
		return false;
		}
	else
		cout << "Decryption passed\n";
	return true;
	} // RandomTest

static __declspec(naked) __int64 GetCounter()
	{ // read time stamp counter in Pentium class CPUs
	_asm
		{
		_emit 0fh
		_emit 31h
		ret;
		}
	} // 

void Timing(int rounds, int keylen, int blocklen)
	{

	__int64 start1, end1, start2, end2, overhead;
	unsigned char key[32],plaintext[32],ciphertext[32];

	int pos;

	CRYPT crypt;
	crypt.SetParameters(keylen*8,blocklen*8);

	srand(0); // make repeatable
	for (pos = 0; pos < keylen; pos++)
		key[pos] = rand();
	for (pos = 0; pos < blocklen; pos++)
		plaintext[pos] = rand();

	// find overhead for these
	start1 = GetCounter();
	end1   = GetCounter();
	overhead = end1 - start1;

	crypt.StartEncryption(key);
	unsigned long min_e = 1000000;
	double total_e = 0;
	for (pos = 0; pos < rounds; pos++)
		{
#ifdef _GPU
		//printf(".");
#endif
		start1 = GetCounter();
		crypt.EncryptBlock(plaintext,ciphertext);
		end1   = GetCounter();
		total_e += end1-start1-overhead;
		if (min_e > (end1-start1-overhead))
			min_e = static_cast<unsigned long>(end1-start1-overhead);
		}

	cout << "Min cycles per encryption (key,block): (" << keylen*8 << ',' << blocklen*8 << ") ";
	cout <<  min_e << endl;

	cout << "Avg cycles per encryption (key,block): (" << keylen*8 << ',' << blocklen*8 << ") ";
	cout <<  total_e/rounds << endl;


	crypt.StartDecryption(key);
	unsigned long min_d = 1000000;
	double total_d = 0;

	for (pos = 0; pos < rounds; pos++)
		{
#ifdef _GPU
		//printf(".");
#endif
		start2 = GetCounter();
		crypt.DecryptBlock(plaintext,ciphertext);
		end2   = GetCounter();
		total_d += end2-start2-overhead;
		if (min_d > (end2-start2-overhead))
			min_d = static_cast<unsigned long>(end2-start2-overhead);
		}

	cout << "Min cycles per decryption (key,block): (" << keylen*8 << ',' << blocklen*8 << ") ";
	cout <<  min_d << endl;
	cout << "Avg cycles per decryption (key,block): (" << keylen*8 << ',' << blocklen*8 << ") ";
	cout <<  total_d/rounds << endl;
	} // Timing


// test a file encryption
void AESEncryptFile(string path)
	{

	#define CPU_CRYPT Rijndael
	#define GPU_CRYPT Rijndael_GPU

	string ext = path.substr(path.find_last_of(".") + 1);
	string fname;
	string dir = "";
	size_t pos = path.find_last_of("\\");
	if(pos != std::string::npos){
		fname.assign(path.begin() + pos + 1, path.end() - ext.length() - 1);
		dir.assign(path.begin(),path.begin() + pos + 1);
	} else {
		fname.assign(path.begin(), path.end() - ext.length() - 1);
	}
	string outFileCPU = dir + fname + "_cpu_out." + ext;
	string outFileGPU = dir + fname + "_gpu_out." + ext;

	ifstream ifile(path,ios_base::binary);
	ofstream ofileCPU(outFileCPU,ios_base::binary);
	ofstream ofileGPU(outFileGPU,ios_base::binary);

	// get file size
	ifile.seekg(0,ios_base::end);
	int size,fsize = ifile.tellg();
	ifile.seekg(0,ios_base::beg);

	// round up (ignore pad for here)
	size = (fsize+15)&(~15);

	char * ibuffer = new char[size];
	char * obuffer = new char[size];
	ifile.read(ibuffer,fsize);

	GPU_CRYPT gpu_crypt;
	CPU_CRYPT cpu_crypt;

	gpu_crypt.SetParameters(192);
	cpu_crypt.SetParameters(192);
	// random key good enough
	unsigned char key[192/8];
	for (int pos = 0; pos < sizeof(key); ++pos)
		key[pos] = rand();
	gpu_crypt.StartEncryption(key);
	cpu_crypt.StartEncryption(key);
	gpu_crypt.Encrypt(reinterpret_cast<const unsigned char*>(ibuffer),reinterpret_cast<unsigned char*>(obuffer),size/16,static_cast<GPU_CRYPT::BlockMode>(0));

	// Gadi:
	INT64 start, finish;
	printf("CPU decryption:\n");
	start = GetTimeMs64();
	cpu_crypt.Decrypt(reinterpret_cast<const unsigned char*>(obuffer),reinterpret_cast<unsigned char*>(ibuffer),size/16,static_cast<CPU_CRYPT::BlockMode>(0));
	finish = GetTimeMs64();
	ofileCPU.write(ibuffer,size);
	printf("CPU processing time: %d (ms)\n",( (finish - start) ));
	printf("GPU decryption:\n");
	gpu_crypt.Decrypt(reinterpret_cast<const unsigned char*>(obuffer),reinterpret_cast<unsigned char*>(ibuffer),size/16,static_cast<CRYPT::BlockMode>(0));
	ofileGPU.write(ibuffer,size);
	

	delete [] ibuffer;
	delete [] obuffer;

	ofileCPU.close();
	ofileGPU.close();
	ifile.close();
	} // AESEncryptFile
////////////////////////////////////////////////////////////////////////////////
// AES main
////////////////////////////////////////////////////////////////////////////////
int aes_main(void)
{
	#ifdef _WIN32
	// to try to prevent windows from interfering too much
	SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
	SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_TIME_CRITICAL);
	#endif
	
	bool allPassed = true; // asusme this

	// some test vectors to check integrity
	allPassed &= TestVector(vectors[0],true);
	allPassed &= TestVector(vectors[1],false);
	allPassed &= TestVector(vectors[2],false);
	allPassed &= TestVector(vectors[3],false);
	allPassed &= TestVector(vectors[4],false);

#ifdef __TimingTests
	// check a bunch of timings for different key and block sizes	
	//for (int block = 16; block <= 32; block += 8)
		for (int key = 16; key <= 32; key += 8)
			Timing(100000,key,16);
#endif

	// this is to randomly test data
	srand(0); // make reproducible
	cout << "Random tests:\n";
	for (int pos = 0; pos < RANDOM_TEST_COUNT; pos++)
		{
		bool passed = RandomTest(pos);
		allPassed &= passed;
		if (passed == false)
			cerr << "Random Test " << pos << " failed\n";
		}

	if (false == allPassed)
		cerr << "ERROR: Some test(s) failed\n";
	else
		cout << "PASSED: All tests passed\n";

	// test a file encryption
	cout << "encrypting file test:\n";
	AESEncryptFile("d:\\test.jpg");

	cout << "\n\n";
	return 0;
} // aes_main
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

	aes_main();

	/*
	shrQAStart(argc, argv);

    printf("> runTest<float,32>\n");
    runTest<float>( argc, argv, 32);
    printf("> runTest<int,64>\n");
    runTest<int>( argc, argv, 64);

    printf("\n[simpleTemplates] -> Test Results: %d Failures\n", g_TotalFailures);
	cutilDeviceReset();

    shrQAFinishExit(argc, (const char **)argv, (g_TotalFailures == 0) ? QA_PASSED : QA_FAILED);
	*/

}

// To completely templatize runTest (below) with cutil, we need to use 
// template specialization to wrap up CUTIL's array comparison and file writing
// functions for different types.  

// Here's the generic wrapper for cutCompare*
template<class T>
class ArrayComparator
{
public:
    CUTBoolean compare( const T* reference, T* data, unsigned int len)
    {
        fprintf(stderr, "Error: no comparison function implemented for this type\n");
        return CUTFalse;
    }
};

// Here's the specialization for ints:
template<>
class ArrayComparator<int>
{
public:
    CUTBoolean compare( const int* reference, int* data, unsigned int len)
    {
        return cutComparei(reference, data, len);
    }
};

// Here's the specialization for floats:
template<>
class ArrayComparator<float>
{
public:
    CUTBoolean compare( const float* reference, float* data, unsigned int len)
    {
        return cutComparef(reference, data, len);
    }
};

// Here's the generic wrapper for cutWriteFile*
template<class T>
class ArrayFileWriter
{
public:
    CUTBoolean write(const char* filename, T* data, unsigned int len, float epsilon)
    {
        fprintf(stderr, "Error: no file write function implemented for this type\n");
        return CUTFalse;
    }
};

// Here's the specialization for ints:
template<>
class ArrayFileWriter<int>
{
public:
    CUTBoolean write(const char* filename, int* data, unsigned int len, float epsilon)
    {
        return cutWriteFilei(filename, data, len, epsilon != 0);
    }
};

// Here's the specialization for floats:
template<>
class ArrayFileWriter<float>
{
public:
    CUTBoolean write(const char* filename, float* data, unsigned int len, float epsilon)
    {
        return cutWriteFilef(filename, data, len, epsilon);
    }
};




////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
template<class T>
void runTest( int argc, char** argv, int len) 
{
    int devID;
    cudaDeviceProp deviceProps;

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
	    devID = cutilDeviceInit(argc, argv);
        if (devID < 0) {
            printf("No CUDA Capable devices found, exiting...\n");
            shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
        }
	}
	else {
	    devID = cutGetMaxGflopsDeviceId();
	    cudaSetDevice( devID );
	}
		
    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    unsigned int num_threads = len;
    unsigned int mem_size = sizeof( float) * num_threads;

    // allocate host memory
    T* h_idata = (T*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (T) i;
    }

    // allocate device memory
    T* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    T* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

	printf("launching kernel...\n");
    // execute the kernel
    testKernel<T><<< grid, threads, mem_size >>>( d_idata, d_odata);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // allocate mem for the result on host side
    T* h_odata = (T*) malloc( mem_size);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, sizeof(T) * num_threads,
                                cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // compute reference solution
    T* reference = (T*) malloc( mem_size);
    computeGold<T>( reference, h_idata, num_threads);

    ArrayComparator<T> comparator;
    ArrayFileWriter<T> writer;

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( writer.write( "./data/regression.dat",
                                     h_odata, num_threads, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res = comparator.compare( reference, h_odata, num_threads);
        printf( "Compare %s\n\n", (1 == res) ? "OK" : "MISMATCH");
        g_TotalFailures += (1 != res);
    }

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cutilDeviceReset();
}
