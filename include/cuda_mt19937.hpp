#include <cstdint>

#ifndef _CUDA_MT19937_HPP_
#define _CUDA_MT19937_HPP_

/**
* +----------------------------------------+
* | @author Ziyao Xiao                     |
* | @description mt19937 algorithm in cuda |
* +----------------------------------------+
*/

namespace fhdnet
{
    namespace __temp_random
    {
        __global__ void __device_initialize(uint32_t *__seed, uint32_t *mt, uint32_t *index);
    }
    class cuda_mt19937
    {
    private:
        uint32_t *mt;

        uint32_t *index;

        __device__ void twist();
    public:
        cuda_mt19937(uint32_t __seed = 0);

        ~cuda_mt19937();

        __device__ uint32_t operator()();

        cuda_mt19937* create_device_ptr();

        void free_device_ptr(cuda_mt19937 *ptr);
    };
}

#endif