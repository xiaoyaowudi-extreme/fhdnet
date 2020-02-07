#include <cstdint>
#include <cuda.hpp>
#include <cuda_mt19937.hpp>

/**
* +----------------------------------------+
* | @author Ziyao Xiao                     |
* | @description mt19937 algorithm in cuda |
* +----------------------------------------+
*/

namespace fhdnet
{
    #define N 624
    #define M 397
    #define R 31
    #define A 0x9908B0DF
    #define F 1812433253
    #define U 11
    #define S 7
    #define B 0x9D2C5680
    #define T 15
    #define C 0xEFC60000
    #define L 18
    #define MASK_LOWER ((1ull << R) - 1)
    #define MASK_UPPER (1ull << R)

    __global__ void __temp_random::__device_initialize(uint32_t *__seed, uint32_t *mt, uint32_t *index)
    {
        uint32_t  i;
        mt[0] = *__seed;
        for (i = 1; i < N; ++i)
        {
            mt[i] = (F * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i);
        }
        *index = N;
    }

    cuda_mt19937::cuda_mt19937(uint32_t __seed)
    {
        CHECK( cudaSetDevice(0) );
        CHECK( cudaMalloc((void**)&mt, N * sizeof(uint32_t)) );
        CHECK( cudaMalloc((void**)&index, sizeof(uint32_t)) );
        uint32_t *__device_seed;
        CHECK( cudaMalloc((void**)&__device_seed, sizeof(uint32_t)) );
        CHECK( cudaMemcpy(__device_seed, &__seed, sizeof(uint32_t), cudaMemcpyHostToDevice) );
        __temp_random::__device_initialize <<< 1, 1 >>> (__device_seed, mt, index);
        CHECK( cudaGetLastError() );
        CHECK( cudaDeviceSynchronize() );
    }
    
    __device__ void cuda_mt19937::twist()
    {
        uint32_t  i, x, xA;
        for (i = 0; i < N; ++i)
        {
            x = (mt[i] & MASK_UPPER) + (mt[(i + 1) % N] & MASK_LOWER);
            xA = x >> 1;
            if (x & 0x1)
                xA ^= A;
            mt[i] = mt[(i + M) % N] ^ xA;
        }
        *index = 0;
    }
    __device__ uint32_t cuda_mt19937::operator()()
    {
        uint32_t  y;
        uint32_t i = *index;
        if ((*index) >= N)
        {
            twist();
            i = *index;
        }
        y = mt[i];
        *index = i + 1;
        y ^= (mt[i] >> U);
        y ^= (y << S) & B;
        y ^= (y << T) & C;
        y ^= (y >> L);
        return y;
    }
    cuda_mt19937* cuda_mt19937::create_device_ptr()
    {
        cuda_mt19937 *ptr;
        CHECK( cudaMalloc((void**)&ptr, sizeof(cuda_mt19937)) );
        CHECK( cudaMemcpy(ptr, this, sizeof(cuda_mt19937), cudaMemcpyHostToDevice) );
        return ptr;
    }
    void cuda_mt19937::free_device_ptr(cuda_mt19937 *ptr)
    {
        CHECK( cudaFree(ptr) );
    }
    cuda_mt19937::~cuda_mt19937()
    {
        CHECK( cudaFree(mt) );
        CHECK( cudaFree(index) );
    }

    #undef N
    #undef M
    #undef R
    #undef A
    #undef F
    #undef U
    #undef S
    #undef B
    #undef T
    #undef C
    #undef L
    #undef MASK_LOWER
    #undef MASK_UPPER
}
